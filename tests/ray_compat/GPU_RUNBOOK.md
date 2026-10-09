# Running vLLM on charm4py.ray on a Linux GPU machine

What was verified on the Mac (CPU, with the `vllm_plugin` test platform) and
what is left for GPUs. Run the steps in order; each one isolates a layer.

## 0. Versions

- vLLM `7fe8fc8` (the audited snapshot), Ray `2.56.1` for the oracle.
- Charm++ branch `charm4py-reconverse-fixes` (based on `reviewed-with-reconverse`;
  Reconverse submodule at main `953eee9`), charm4py branch `ray-dev`.
- Python 3.10-3.14; one venv for everything is fine on Linux (the Mac used two).

## 1. Build

```bash
# Charm++ + Reconverse with the charm4py target. No tracing for performance runs
# (a tracing build writes Projections logs next to the launcher).
cd charm && ./build charm4py reconverse-linux-x86_64 --with-production --suffix=charm4py \
    --disable-fortran -j     # reconverse-linux-arm8 on Grace (GH200/GB200)

python -m venv ~/venvs/vllm_c4p && . ~/venvs/vllm_c4p/bin/activate
# vLLM publishes per-commit wheels (wheels.vllm.ai/<commit>); building from source takes long
pip install "vllm @ git+https://github.com/vllm-project/vllm@7fe8fc8"
pip install "ray==2.56.1" greenlet dill cython pytest tblib            # ray: oracle only
cd charm4py && CHARM4PY_CHARM_DIR=<charm> CHARM4PY_BUILD_TRIPLET=reconverse-linux-x86_64 \
    CHARM4PY_CHARM_BUILD_DIR=<charm>/reconverse-linux-x86_64-charm4py \
    pip install --no-build-isolation -e .
export LCRUN=<charm>/reconverse-linux-x86_64-charm4py/_deps/lci-src/lcrun
```

## 2. Layer checks (CPU only, minutes)

```bash
python -m charmrun.start +p4 -m charm4py.ray.launch tests/ray_compat/contract.py      # 58/58
python tests/ray_compat/contract.py                                                   # oracle
python -m charmrun.start +p4 -m charm4py.ray.launch tests/ray_compat/engine_process.py spawn
python -m charmrun.start +p8 -m charm4py.ray.launch tests/ray_compat/refcount_stress.py
```

Then check GPU discovery: on a node with N GPUs, `ray.cluster_resources()['GPU']`
under the launcher must be N without `--num-gpus` (detected from
`CUDA_VISIBLE_DEVICES`, NVML or `nvidia-smi`), and the device IDs must be the
physical ones when `CUDA_VISIBLE_DEVICES` restricts them.

## 3. vLLM on GPUs, one node

```bash
# no plugin, no --num-gpus: the real CUDA platform and detected GPUs
G=tests/ray_compat/vllm_generate.py
python $G 1 1 mp                                                         # reference
python -m charmrun.start +p4 -m charm4py.ray.launch $G 1                 # TP=1
python -m charmrun.start +p4 -m charm4py.ray.launch $G 2                 # TP=2 (NCCL)
python -m charmrun.start +p4 -m charm4py.ray.launch $G 1 2               # PP=2
ray start --head; RAY_ADDRESS=auto python $G 2; ray stop                  # oracle
# vLLM's own executor tests (they assume 4 GPUs)
cd vllm && python -m charmrun.start +p4 -m charm4py.ray.launch -m pytest \
    tests/distributed/test_ray_v2_executor.py
```

Expect identical text to the mp executor. What GPUs add over the Mac runs:
`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` (vLLM sets it for its workers, so
charm4py.ray must not set `CUDA_VISIBLE_DEVICES` and vLLM maps physical IDs
itself), NCCL between actor processes, CUDA initialized in the driver (vLLM then
spawns its engine process instead of forking it), and GPU memory per actor.
PP=2 failed on CPU in vLLM itself; on GPUs it should work.

Then the plan's exit criteria, with the scripts used on the Mac:

```bash
python -m charmrun.start +p4 -m charm4py.ray.launch tests/ray_compat/vllm_cycles.py 100 2
tests/ray_compat/vllm_serve_dp.sh charm4py      # and: ... ray   (OpenAI server, DP=2)
CHARM4PY_RAY_FAKE_NODES=3 python -m charmrun.start +p6 -m charm4py.ray.launch \
    tests/ray_compat/multinode.py               # control plane across nodes
```

`vllm_cycles.py` checks resources, objects and processes; on GPUs also watch
`nvidia-smi` memory and `/dev/shm` between cycles.

## 4. Multi-node (not tried yet)

`charmrun.start` launches through `lcrun`, which is single-node. Several nodes
need Reconverse started by Slurm with a PMI backend LCI supports (to be worked
out), one host process per PE as now. Known limits: processes outside the job
can connect only on PE 0's node (vLLM's engine process runs there); set
`CHARM4PY_RAY_NODE_IP` (or `VLLM_HOST_IP`) when the default-route address is not
the one the nodes reach each other by.

## What is measured where

- Correctness: text identical to mp and to Ray; vLLM's executor tests.
- Overheads that matter here: `LLM()` startup (actor creation, placement group,
  four rounds of actor calls), worker-death detection latency (`ray.wait` on the
  run refs), shutdown time. Token generation does not go through Ray.

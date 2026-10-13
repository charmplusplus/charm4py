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

Linux issues the Mac never showed (found on Delta, glibc 2.34, GCC 14):

- `libcharm.so` used to be built with `-ftls-model=initial-exec`, which pulls
  libreconverse's thread-locals (about 23 KB, mostly `conds` in
  `conv-conds.cpp`) into the static TLS block. Python loads it with `dlopen`,
  and glibc's default reserve is too small: `libreconverse.so: cannot allocate
  memory in static TLS block`, during `pip install` (setup.py loads it) and at
  run time. Charm++ now leaves that flag out for the charm4py target
  (`cmake/detect-features-cxx.cmake`); with an older Charm++ build, export
  `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=65536` instead. After changing
  the flag, rebuild from clean (`make clean` in the build directory): targets
  compiled through `charmc` read its options from `conv-mach-opt.sh`, which
  make does not track, and stale objects keep the initial-exec relocations.
  `readelf -d libcharm.so` must show no `STATIC_TLS` flag.
- GCC 14 makes charmlib's pointer-type mismatches errors. Build with
  `CC=gcc CXX=g++ CFLAGS="-Wno-error=incompatible-pointer-types -Wno-error=implicit-function-declaration"`
  (on Cray systems the `cc` wrapper also works with these flags).
- Keep the venv off slow parallel filesystems: torch and vLLM import tens of
  thousands of files in every PE and worker process. On Delta, a venv on
  HDD-backed Lustre did not finish `import numpy` in 280 s on a GPU node; pack
  the env into a tarball and unpack it to node-local `/tmp` at job start
  (about 11 s for 8 GB).
- An aborting PE can hang the job and writes a core file (about 1 GB each) in
  the working directory; use `ulimit -c 0` in batch scripts.

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

Results on Delta, one node with 4 A40s (2026-10-13; vLLM `7fe8fc8` wheel,
torch 2.13 / CUDA 13.0, Charm++ `charm4py-reconverse-fixes` with Reconverse
main `988a8b4`, `+p4`):

| Check | Result |
|---|---|
| layer checks (contract at `+p1`/`+p4`/`+p8`, multinode, refcount, engine fork/spawn) | all pass |
| GPU discovery (Slurm's `CUDA_VISIBLE_DEVICES`, restricted to `2,3`, unset/NVML) | 4 / 2 / 4 GPUs, physical IDs |
| `vllm_generate.py` TP=1, TP=2 (NCCL), PP=2 | text identical to `mp` |
| startup, warm caches: TP=1 / TP=2 / PP=2 | 15.0 / 16.9 / 57.7 s (`mp` TP=2: 9.6 s) |
| `test_ray_v2_executor.py` | 23/23 (the CPU-only PP=2 failures do not occur) |
| `vllm_cycles.py 100 2` | PASS; startup 15.6-16.8 s after a 26.8 s first cycle, PE memory flat, GPUs and `/dev/shm` empty after |
| `vllm_serve_dp.sh charm4py` | PASS |

## 4. Multi-node

`charmrun.start` launches through `lcrun`, which is single-node. Across nodes,
start the launcher with `srun` directly (LCI bootstraps through Slurm's PMI),
one process per PE as before, and pass the total PE count with `+pe`:

```bash
# in an allocation of 3 nodes, 2 tasks and 1 GPU per node; the env must be on every node
export FI_CXI_RX_MATCH_MODE=hybrid       # Slingshot: CXI list-entry exhaustion otherwise
export PMI_MAX_KVS_ENTRIES=1000          # see below
srun --unbuffered --kill-on-bad-exit=1 -N 3 --ntasks-per-node=2 \
    python -m charm4py.ray.launch tests/ray_compat/vllm_generate.py 2 +pe 6
```

- Cray PMI2 allows 24 key-value entries per job by default. LCI before
  `06748025` (allgather bootstrap) publishes too many and every process aborts
  with `_pmi2_add_kvs: The KVS data segment of 24 entries is not large enough`.
  Reconverse main pins LCI `ca88ce2c`, which predates it; raise the limit as
  above, or build with an LCI at `dfb924cf` or later.
- `--unbuffered` matters: buffered output loses the fatal line and a failure
  looks like a hang.
- Do not unset `SLURM_NTASKS` here (single-node lcrun jobs inside an allocation
  must, since LCI then waits for that many processes).

Known limits: processes outside the job can connect only on PE 0's node (vLLM's
engine process runs there); set `CHARM4PY_RAY_NODE_IP` (or `VLLM_HOST_IP`) when
the default-route address is not the one the nodes reach each other by.

Results on Delta, 3 nodes with 1 A40 each, 2 PEs per node (2026-10-13):

| Check | Result |
|---|---|
| `multinode.py` | 8/8 |
| `contract.py` | 53/58: the 5 GPU-accounting tests assume one node (`ray.init(num_gpus=2)` is per node, so the cluster has 6) |
| GPU discovery | 3 GPUs, each actor gets physical ID 0 on its own node |
| `vllm_generate.py` TP=2 across nodes / PP=2 across nodes | text identical to single-node `mp`; startup 29.3 / 14.6 s |
| `vllm_cycles.py 20 2`, TP=2 across nodes | PASS; startup 14.6-14.9 s after a 28.7 s first cycle, all GPUs free and no objects after (the process check sees only the driver's node) |

NCCL used its socket transport (`NET/Socket`, over `eth1`, `hsn0` and
`hsn0.561`); the AWS OFI NCCL plugin (`module load aws-ofi-nccl`) would put it
on Slingshot directly, which matters for performance, not correctness.

## What is measured where

- Correctness: text identical to mp and to Ray; vLLM's executor tests.
- Overheads that matter here: `LLM()` startup (actor creation, placement group,
  four rounds of actor calls), worker-death detection latency (`ray.wait` on the
  run refs), shutdown time. Token generation does not go through Ray.

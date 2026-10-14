# Ray on Charm4py: progress and next steps

Status as of 2026-10-13, branch `ray-dev`. Steps 1-4 of "Next steps" are done
on Delta (NCSA; one node with 4 A40s, and three nodes with one A40 each); step 5
(research measurements) is next. Work is being split across machines because
Delta's GPU queue is slow for us (see "Handing work to another machine").

Read this first in a new session; details are in `tests/ray_compat/README.md`
(how it works, test results) and `tests/ray_compat/GPU_RUNBOOK.md` (build and
check steps for a GPU machine, Linux build issues, multi-node launch under
Slurm, all measured numbers). The research plan is
`vllm_charm4py_research_plan.md` in the repository root.

## Goal

Run unmodified vLLM (Ray executor, `RayExecutorV2`, vLLM `7fe8fc8`) on
`charm4py.ray`, an implementation of the Ray Core API (Ray 2.56.1 semantics) on
Charm4py over Reconverse; then study whether a Charm-native control plane helps
(the research question: placement, resources and failure handling for many
vLLM replicas without a central GCS).

## Where it stands

Everything below was verified on an M-series Mac (CPU only).

- **Ray compatibility**: `tests/ray_compat/contract.py`, run unchanged against
  real Ray and against charm4py.ray, passes 58/58 on both (charm4py at `+p1`,
  `+p2`, `+p4`, `+p8`).
- **vLLM end to end, on CPUs**: a test-only platform plugin
  (`tests/ray_compat/vllm_plugin`, `VLLM_CPU_RAY=1`) makes CPU worker slots look
  like logical GPUs, so the real Ray executor runs without vLLM changes. Greedy
  generation at TP=1 and TP=2 gives the same text as real Ray and as vLLM's
  multiprocessing executor, at the same speed (TP=2: about 8.6 s `LLM()`
  startup, 1.5 s generation). vLLM's own `tests/distributed/test_ray_v2_executor.py`
  passes 21/23 under the launcher; the 2 failures are PP=2 generation, which
  fails the same way with vLLM's multiprocessing executor (a vLLM CPU bug). The
  OpenAI server with DP=2 on the Ray backend works; 25 create/generate/shutdown
  cycles leak nothing and keep startup flat.
- **vLLM on GPUs** (Delta, 4x A40, no plugin): TP=1, TP=2 (NCCL between actor
  processes) and PP=2 give the same text as the `mp` executor; vLLM's executor
  tests pass 23/23; 100 TP=2 cycles leak nothing with flat ~16 s startup; the
  DP=2 OpenAI server works. Numbers and the Linux build issues (static TLS,
  GCC 14, slow filesystems) are in `tests/ray_compat/GPU_RUNBOOK.md`.
- **Multi-node** (Delta, 3 nodes, Reconverse started by `srun`): the
  multi-node control-plane tests pass 8/8 on real nodes, and vLLM TP=2 and
  PP=2 spanning nodes give the same text; 20 cross-node TP=2 cycles leak
  nothing. The launch recipe and a Cray PMI limit are in
  `tests/ray_compat/GPU_RUNBOOK.md` section 4.
- **NCCL network plugin** (Delta, `module load aws-ofi-nccl/1.19.2`): NCCL runs
  over Slingshot (`NET/OFI Selected provider is cxi`), 9.8 GB/s all-reduce bus
  bandwidth between two nodes at 256 MB; vLLM TP=2 across nodes still gives the
  same text, and its startup drops from 34.3 s (sockets) to 15.6 s. Charm4py's
  PEs run unchanged with the plugin's libfabric settings.

## How it works (short)

- `python -m charmrun.start +pN -m charm4py.ray.launch [--num-gpus N] script.py`
  (or `-m module`) installs charm4py.ray as `ray` in every process.
- Charm4py processes (one per PE) are the runtime; user code runs in worker
  processes that do not load Charm++: the driver (served by PE 0), one process
  per actor, and a task-worker pool per PE. Workers talk to their PE over a
  socket; PEs never deserialize user values (`_ipc.Opaque`).
- Processes the job's processes start (vLLM's engine process, forked or
  spawned by the driver) connect through `_Gateway` and run as jobs of their own.
- Control plane (`charm4py/ray/_control.py`), no central scheduler: each PE
  coordinates the actors and placement groups created on it; each node's last
  PE is the only authority for that node's resources and accelerator IDs and
  places actors there; names are hashed over all PEs; placement groups reserve
  in two phases among the nodes they use. Only the job table is on PE 0.
- Reference counting (`_refs.py`): a borrower tree between PEs (no message-order
  assumptions) and exact token counts between a PE and its workers.
- `CHARM4PY_RAY_FAKE_NODES=k` splits one machine's PEs into k logical nodes,
  for testing the multi-node logic.

Main files: `charm4py/ray/api.py`, `_hosts.py` (PE-side chares), `_control.py`,
`_worker_main.py`, `_ipc.py`, `_proc.py`, `_refs.py`, `launch.py`.

## Tests and the results to expect

| Command (under `python -m charmrun.start +pN -m charm4py.ray.launch ...`) | Expect |
|---|---|
| `tests/ray_compat/contract.py` (also with plain `python`, real Ray) | 58/58 |
| `CHARM4PY_RAY_FAKE_NODES=3` ... `+p6` `tests/ray_compat/multinode.py` | 8/8 (Ray: `ray_oracle_cluster()`) |
| `+p8` `tests/ray_compat/refcount_stress.py` | PASS, store ends empty |
| `tests/ray_compat/engine_process.py fork` / `spawn` | PASS |
| `tests/ray_compat/vllm_generate.py TP PP [mp]` | same text as `mp` and Ray |
| `tests/ray_compat/vllm_cycles.py N TP` | PASS, flat startup |
| `tests/ray_compat/vllm_serve_dp.sh charm4py` / `ray` | PASS, same completions |
| `cd vllm; ... --num-gpus 4 -m pytest tests/distributed/test_ray_v2_executor.py` | 21/23 on CPU |

The Charm4py regression set (examples and `tests/`: futures, pool, channels,
...) also passes; run it with `charmrun.start +p4` per file.

## Handing work to another machine

### What the code depends on, and what is not pushed yet

- charm4py: branch `ray-dev`; pull before starting (commits made on Delta are
  pushed by hand, so check that this file's latest changes are there).
- Charm++: branch `charm4py-reconverse-fixes` of charmplusplus/charm at
  `bc6d7a283` or later. That commit moves the Reconverse submodule to main
  `988a8b4` and builds the charm4py target without `-ftls-model=initial-exec`;
  without it, on Linux, Python cannot load `libcharm.so` ("cannot allocate
  memory in static TLS block"). Run `git submodule update --init` after
  checking it out. A build directory configured before that commit must be
  rebuilt from clean; `readelf -d libcharm.so` must show no `STATIC_TLS`
  (runbook, section 1). macOS is not affected.
- vLLM `7fe8fc8`: install the per-commit wheel (runbook, section 1), Ray
  `2.56.1`, Python 3.12. The runbook lists the GCC 14 flags charm4py needs.

### Set up on a new Linux machine

Follow `tests/ray_compat/GPU_RUNBOOK.md` sections 0-2, then run the layer
checks (contract 58/58 at `+p4`; multinode with fake nodes 8/8). On a GPU node
continue with section 3; expected numbers are in its results table. Keep the
venv on a local disk: torch and vLLM import tens of thousands of files in every
process (on Delta's Lustre this took minutes per process).

### Real-Ray baseline (done 2026-10-13)

A Ray 2.56.1 cluster started inside the Slurm allocation runs vLLM TP=2 and
PP=2 across three nodes with the same text as charm4py.ray; the recipe and
results are in the runbook, section 4. Startup times so far are single runs
and not a comparison. `multinode.py` passes 7/8 on real Ray: one test expects
more locality than Ray guarantees (runbook).

Delta's job scripts are in `/work/nvme/mzu/rao1/vllm_c4p/runs/` (environment
tarball staged to node-local `/tmp`, `step3_vllm_gpu.sh`,
`step4_multinode.sh`, `step5_pre.sh`); they are site-specific and not in this
repository.

### Step 5: status and who runs what (2026-10-14)

Order (decided 2026-10-13): H4 (steady-state overhead, the control) first, then
H1 (placement and startup at scale) and H3 (failure detection and recovery) in
parallel; H2 is deferred. H1 and H3 use model-free replicas on logical GPUs.
H3 measures recovery after a process crash; a dead PE ends the whole job (see
below), and whether to tolerate node failures is not decided.

**H4 is done** (Delta, one A40x4 node, Qwen/Qwen3-8B at TP=2, `vllm bench
serve`, 400 random prompts of 512 in / 128 out, one repetition per point):

| request rate | backend | req/s | median TTFT | median TPOT |
|---|---|---|---|---|
| 2/s | mp / ray / c4p | 1.98 / 1.98 / 1.98 | 96.1 / 97.3 / 96.8 ms | 17.95 / 17.97 / 17.99 ms |
| 8/s | mp / ray / c4p | 7.64 / 7.64 / 7.64 | 58.1 / 58.7 / 58.4 ms | 19.72 / 19.73 / 19.74 ms |
| unlimited | mp / ray / c4p | 26.94 / 26.84 / 26.83 | 1000 / 1173 / 1159 ms | 67.81 / 67.77 / 67.75 ms |

charm4py.ray costs nothing measurable in steady state, as expected. At the
unlimited rate the median TTFT is queueing (P99 is about 9.8 s on all three).

**H1 and H3 move to Stampede3.** The harness is
`tests/ray_compat/replica_bench.py` (`bench/replicas.py`, Delta's first
version, is removed). One node, 4 fake nodes:

    CHARM4PY_RAY_FAKE_NODES=4 python -m charmrun.start +p8 -m charm4py.ray.launch \
        --num-gpus 4 tests/ray_compat/replica_bench.py --replicas 8 --tp 2 --rounds 3
    ray start --head --num-cpus 8 --num-gpus 16
    RAY_ADDRESS=auto python tests/ray_compat/replica_bench.py --replicas 8 --tp 2 --rounds 3

For H3 add `--kill --kill-mode sigkill --kill-count 2 --recover`: each round,
2 replicas lose a worker process to SIGKILL (sent by a helper actor on the
worker's node, as a crash) and start again; the records gain `detect` (kill
to the monitor's `ray.wait` returning), `recover` (kill to serving again) and
`restart_pg/actors/init/run`. `--kill` alone uses `ray.kill`. Across nodes,
start charm4py.ray with `srun --kill-on-bad-exit=1` and the total `+pe`
(runbook section 4), and Ray as a cluster inside the allocation, one raylet
per node with logical `--num-gpus`.

What Delta found that applies there:

- **Fail fast**: `charmrun.start` now starts the Reconverse processes itself
  and, when one exits nonzero, stops the rest and returns its code (`lcrun`
  hung instead). A PE killed under `failure_probe.py --kill pe` ends the job in
  3 s with exit 137. Across nodes, `srun --kill-on-bad-exit=1` does the same.
- **First H3 numbers** (one Delta GPU node used as a CPU node, 8 replicas x
  tp=2 on 16 logical GPUs, 2 crashed per round, 2 rounds, p50; charm4py.ray on
  4 fake nodes, Ray on one node; 0 errors on both):

  | | detect | recover | restart_pg | restart_actors |
  |---|---|---|---|---|
  | charm4py.ray | 4 ms | 115 ms | 4 ms | 94 ms |
  | Ray 2.56.1 | 8-9 ms | 1.22-1.35 s | 0.39-0.52 s | 1.21-1.33 s |

  Single runs at one small size; recovery is mostly creating the new worker
  actors.
- **First H1 look** (same runs, not yet a comparison): `actors` p50 0.19 s on
  charm4py.ray against 1.6-2.3 s on Ray. Check how worker processes are
  started on each side before reading anything into it.
- **Ray with logical GPUs on a GPU node** refuses to start ("Attempting to start
  raylet with 16 GPU, but CUDA_VISIBLE_DEVICES contains [0..3]"): unset
  `CUDA_VISIBLE_DEVICES` first.
- **Ray hang on Delta, not reproduced**: with `bench/replicas.py` on a 4-node
  Ray cluster of CPU nodes, the 32 replica managers were created and the run
  then made no progress for 15 minutes. `replica_bench.py` on a 2-node Ray
  cluster (GPU nodes, 32 logical GPUs, 8 replicas) ran without errors. The
  cause was not found; watch for it in the first multi-node Ray runs.
- **Reconverse's LCI pin**: Reconverse main fetches LCI `ca88ce2c`, which
  predates LCI's allgather bootstrap (`06748025`, `dfb924cf`): each process
  publishes one PMI key per process per round, so Cray PMI's KVS needs
  `PMI_MAX_KVS_ENTRIES` to grow with the square of the process count. Delta
  needed 100000 at 32 processes on 4 nodes (1000 was too small). Moving the
  pin removes that. Kept as an environment variable for now (decided
  2026-10-13).

Delta keeps the GPU runs: vLLM DP serving over 8 and 16 A40x4 nodes
(preemptible, queued) with opt-125m and Qwen3-8B on Ray and charm4py.ray.

Still open, any machine with 2 GPUs: **single-node PP=2 startup** takes 57.7 s
against 15.0-16.9 s for TP=1/TP=2 on one A40 node, but 14.6 s for PP=2 across
nodes.

## Next steps

In this order (`tests/ray_compat/GPU_RUNBOOK.md` has the commands). Steps
1-4 are done (2026-10-13); comparisons with real Ray were left for step 5.

1. Build Charm++ (branch `charm4py-reconverse-fixes`, a `reconverse-linux-*`
   charm4py build, no tracing) and charm4py; run the contract suite, the
   multi-node test with fake nodes and the refcount stress test on CPU.
2. Check GPU discovery: with no `--num-gpus`, `ray.cluster_resources()['GPU']`
   must equal the node's GPU count, and device IDs must follow
   `CUDA_VISIBLE_DEVICES`.
3. vLLM on one GPU node, without the plugin: `vllm_generate.py` at TP=1, TP=2
   (NCCL), PP=2, against the `mp` executor and real Ray; then vLLM's executor
   tests, `vllm_cycles.py 100 2` (also watch `nvidia-smi` and `/dev/shm`), and
   `vllm_serve_dp.sh`.
4. Multi-node: start Reconverse across nodes under Slurm with `srun` directly
   (`lcrun` is single-node; runbook section 4), then TP and PP across nodes.
5. Research measurements: placement/startup latency and failure-detection time
   with many concurrent vLLM replicas, distributed control plane vs. real Ray.
   The centralized `_RayControl` was removed in `2348810`; it is in git history
   if a centralized baseline inside charm4py is wanted.

Gaps that may matter there (none is on vLLM's default path):

- Tasks are placed round-robin with no resource admission; tasks with
  `num_gpus` get no device IDs or `CUDA_VISIBLE_DEVICES`.
- No restarts or retries and no recovery from a dead PE or node: the launcher
  stops the whole job (`charmrun.start`, or `srun --kill-on-bad-exit=1`). No
  `ray.cancel`, no `max_concurrency > 1` or async actors.
- Processes outside the job can connect only on PE 0's node.
- Each idle PE uses 10-18% of a core (polling).

## Gotchas

- vLLM renames its worker processes (`VLLM::Worker`, `VLLM::DPCoordinator`):
  `pkill -f _worker_main` misses them, and orphans skew later timings. Clean up
  with `pkill -f VLLM::` too.
- On a machine without GPUs, real Ray's workers inherit the environment of the
  `ray start` daemon: start it with `VLLM_CPU_RAY=1` already set.
- Ray-backend DP on one machine needs `VLLM_HOST_IP=127.0.0.1` on both backends
  (vLLM takes the LAN address as the DP master; a local Ray node is 127.0.0.1).
  `vllm_serve_dp.sh` now sets it unless it is already set.
- On the CPU backend: `VLLM_CPU_KVCACHE_SPACE` limits memory use;
  `VLLM_CPU_OMP_THREADS_BIND=nobind` is needed for TP=2 with the mp executor.
- Tracing builds of Charm++ write Projections files (`.sts`, `.sum`,
  `.log.gz`, `.projrc`) next to the script or `charm4py/ray/launch.py`; use a
  build without tracing for runs, and delete them before committing.
- After changing a `.pyx` file, reinstall charm4py
  (`pip install --no-build-isolation -e .` with `CHARM4PY_CHARM_DIR`,
  `CHARM4PY_BUILD_TRIPLET`, `CHARM4PY_CHARM_BUILD_DIR` set).
- `charmrun.start` starts the job in its own process group and forwards SIGINT
  and SIGTERM to it; send signals to `charmrun.start`, not to individual PEs.

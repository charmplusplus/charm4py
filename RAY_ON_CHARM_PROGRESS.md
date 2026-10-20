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

### Step 5 on Stampede3: H1 and H3 results (2026-10-15)

Setup: TACC Stampede3 SKX nodes (48 cores, Omni-Path), charm4py.ray at 4 PEs
per node over LCI with libfabric `verbs;ofi_rxm` (the only LCI provider that
works on Omni-Path; runbook, "Omni-Path (Stampede3)"), Ray 2.56.1 as a cluster
inside the same allocation, both backends in the same job on the same nodes.
`replica_bench.py --tp 2 --rounds 6 --kill --kill-mode sigkill --kill-count 2
--recover`, 8 logical GPUs per node (4 replicas per node), venv staged to
node-local `/tmp`. One job per node count, 5 measured rounds each; 0 errors on
every run. An H3 run also records the H1 startup phases, so one sweep gives
both. charm4py at `c90332f` (C pickler, below); at 64 nodes the two charm4py
runs of the job are given as a range. p50:

| nodes | replicas | makespan c4p / Ray | pg c4p / Ray | actors c4p / Ray | detect c4p / Ray | recover c4p / Ray | restart_pg c4p / Ray |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.44 / 2.43 s | 4 / 4 ms | 0.43 / 2.42 s | 4 / 8 ms | 0.44 / 2.48 s | 2 / 168 ms |
| 2 | 8 | 0.47 / 2.48 s | 9 / 7 ms | 0.44 / 2.43 s | 3 / 7 ms | 0.44 / 2.48 s | 2 / 174 ms |
| 4 | 16 | 0.50 / 2.68 s | 18 / 14 ms | 0.47 / 2.45 s | 4 / 7 ms | 0.44 / 3.21 s | 3 / 746 ms |
| 8 | 32 | 0.59 / 2.69 s | 41 / 30 ms | 0.52 / 2.48 s | 4 / 7 ms | 0.44 / 3.13 s | 3 / 875 ms |
| 16 | 64 | 0.76 / 2.71 s | 53 / 59 ms | 0.61 / 2.49 s | 5 / 7 ms | 0.46 / 3.06 s | 4 / 781 ms |
| 32 | 128 | 1.05 / 2.79 s | 111 / 128 ms | 0.76 / 2.54 s | 5 / 8 ms | 0.45 / 2.77 s | 4 / 426 ms |
| 64 | 256 | 1.73 / 4.38 s | 320 / 293 ms | 1.18-1.19 / 3.83 s | 7 / 10 ms | 0.46 / 2.87 s | 5-6 / 554 ms |

Before `c90332f` (2026-10-14, same setup) charm4py's makespan was 0.47, 0.51,
0.60, 0.79, 1.03, 1.69 and 3.10 s at 1-64 nodes, and its placement groups took
545 ms at 64 nodes; Ray's numbers were the same within run-to-run variation.

What the numbers say:

- **Startup (H1)**: charm4py.ray reaches serving first at every size: 5.5x
  ahead of Ray at 1 node, 2.5x at 64. Most of the lead is worker-process
  startup; placement groups now take about as long on both.
- **The startup gap is like for like.** charm4py.ray never starts workers
  ahead of time (one `subprocess.Popen` per actor, `_proc.py`). Ray prestarts
  about one idle worker per CPU, and an actor without a `runtime_env` reuses
  one (10-30 ms), but an actor with one gets a new process (2.0-2.1 s on SKX;
  `import ray` alone is 1.0 s, against 0.4 s for charm4py's worker). vLLM's
  `RayExecutorV2._build_runtime_env` always sets `env_vars` on CUDA and ROCm
  (`ray_noset_device_env_vars`), so vLLM's workers, like the harness's, never
  reuse a prestarted Ray worker.
- **Crash recovery (H3)**: detection is close (3-7 ms against 7-10 ms).
  charm4py.ray is serving again in 0.44-0.46 s at every size; Ray takes
  2.5-3.2 s, of which 0.2-0.9 s is re-reserving the placement group right after
  the teardown, and the rest is starting the new workers.
- **charm4py.ray's own scaling cost**: the `init` phase (three rounds of actor
  calls per replica, made from the driver's threads) grows with the number of
  replicas, not nodes (0.27 s for 128 replicas on 4 nodes, 0.33 s for 128 on
  32, before `c90332f`); Ray's stays near 3 ms. Every call from the driver
  process is two synchronous round trips through PE 0 (`.remote()` is
  `_client.call('_invoke')`, `ray.get` is `_client.call('_get_stored')`), where
  Ray's `.remote()` returns at once and the call goes straight to the actor.
  py-spy at 128 replicas on 4 nodes, with `+CmiSleepOnIdle` so that idle time
  is not counted as busy: PE 0 was 66-97% busy during startup, and 13-17% of
  its samples were dill pickling the Charm messages that forward the calls
  (dill's `Pickler` is Python's pure-Python one). `c90332f` pickles messages
  with the C pickler and gives dill only what needs it; at 64 nodes the `init`
  phase went from 0.75 s to 0.39 s and placement groups from 545 ms to 320 ms.
  What remains is PE 0 serving every call of the driver. Making the driver's
  `.remote()` not wait for PE 0 (spare references sent ahead by the host) was
  tried and did not help: 1.76-1.81 s makespan at 64 nodes against 1.73 s
  without it, and no change at 128 replicas on 4 nodes. Not committed; the
  patch is on Stampede3 at `~/c4p_runs/async_invoke_and_diag.patch`.
- **The PE 0 funnel is the harness's, not vLLM's.** `replica_bench.py` runs
  every replica's executor in a driver thread, so PE 0 serves all of their
  calls. vLLM with data parallelism on Ray does not: `CoreEngineActorManager`
  reserves a node-pinned `STRICT_PACK` group per rank and starts an engine
  actor in it, and the engine builds the executor, so the executor's calls are
  served by the engine's own PE. `--engine-actors` lays the benchmark out that
  way (H3, 256 replicas on 16 nodes of 32 logical GPUs, which reproduces the
  64-node driver-mode numbers: makespan 1.63-1.65 s there against 1.72 s at
  64 nodes). p50:

  | | charm4py driver mode | charm4py engine actors | Ray engine actors |
  |---|---|---|---|
  | makespan | 1.63-1.65 s | 1.33-1.35 s | 8.11 s |
  | workers created (after the engine, or the group) | 0.90 s | 0.47-0.49 s | 3.83 s |
  | startup calls (`init` after `actors`) | 0.36 s | 6 ms | 2 ms |
  | recover | 0.44-0.45 s | 0.86-0.88 s | 5.10 s |

  With engine actors charm4py's worker creation and startup calls cost the
  same at 256 replicas as at 8 (0.44 s and 6 ms on 2 nodes), although the
  engine adds a process start in series (0.43 s on 2 nodes; Ray's 2.3 s);
  what still grows is the driver's own part, 256 groups and engines through
  PE 0 (`pg` 0.009 -> 0.22 s, the engine's start 0.42 -> 0.56 s after it).
  Recovery restarts the whole rank (group, engine, workers), as a restarted
  DP rank would, hence twice driver mode's. Caveat: 32 logical GPUs per node
  put 48 Python processes on each 48-core node; Ray's process starts slow
  down under that (engine 2.3 -> 3.0 s, workers 2.4 -> 3.8 s from 2 nodes of
  8), charm4py's do not, so Ray's 8.11 s overstates what 64 nodes of 8 would
  give. Spreading the driver's calls over its node's PEs (refs and actor-call
  order are per link, so it is not small) would only help single-driver
  workloads; not pursued.
- **Engine actors at 64 nodes** (2026-10-20, job 3620559, tag `n64_eng`, 8
  logical GPUs per node, 256 replicas, charm4py twice and Ray once in the same
  job, 0 errors). p50:

  | | charm4py | Ray |
  |---|---|---|
  | makespan | 1.57 / 1.47 s | 5.59 s |
  | startup (`init`) | 1.38 / 1.35 s | 5.11 s |
  | engines started | 0.91 / 0.88 s | 2.72 s |
  | recover | 0.91 s | 5.54 s |
  | `pg` | 0.36 / 0.35 s (p99 0.39) | 0.38 s (p99 0.86) |

  charm4py is 3.6x faster than Ray in vLLM's DP layout at 512 GPUs, and
  engine actors cut its makespan from 1.73 s (driver mode, `n64_h3fix`) to
  1.47-1.57 s. They make Ray slower (4.38 -> 5.59 s): every replica pays one
  more actor process start. The 16-node proxy gave charm4py's number (1.33 s)
  but not Ray's (8.11 s; see the caveat above), so Ray comparisons need real
  node counts. The first charm4py run took 89 s to its first output because
  the staged venv was read from a cold `$WORK`; the second took 8 s.
- **Delta's multi-node Ray hang** did not occur on Ray clusters of 2-64 nodes.
- **A PE can segfault at shutdown**, after the benchmark has finished and
  printed its summary (1 of the 23 charm4py runs in the two sweeps, at 1
  node; 2 of 82 tests in one run of charm4py's own test suite). This is
  libfabric, not charm4py: every core has the same stack, `ConverseExit` ->
  `CommBackendLCI2::exit` -> `lci::free_device` -> `rxm_ep_close` ->
  `rxm_freeall_conns` -> `rxm_flush_msg_cq` -> `rxm_free_rx_buf` -> `fi_recv`,
  in a single-threaded process after the exit barrier. Two tests looped on
  one node, several runs at a time:

  | libfabric, provider | runs | segfaults at exit |
  |---|---|---|
  | system 2.6.0a1, `verbs;ofi_rxm` | 2990 | 486 (16%) |
  | system 2.6.0a1, `tcp;ofi_rxm` | 1680 | 0 |
  | 2.7.0, `verbs;ofi_rxm` | 2541 | 0 |

  2.7.0's release notes fix "the msg_ep use-after-free at connection
  teardown" in RxM. The 2.7.0 runs include 1128 started 6 at a time, which
  did not hang (3 runs in an earlier job did hang once at startup, with no
  output; not seen again). 2.7.0 is built in `$WORK/libfabric-2.7.0` (only
  the verbs, rxm, tcp and shm providers; `libhwloc.so.15` copied in) and
  selected with `LD_LIBRARY_PATH` plus an empty `FI_PROVIDER_PATH`; LCI loads
  `libfabric.so.1` by soname, so nothing is rebuilt. `step5_h1.slurm` stages it to
  each node's `/tmp` and uses it by default (`LIBFABRIC=sys`, or `+sysfab` on
  a backend, for the system one; read from `$WORK`, the first launch of a job
  stalled for 2+ minutes). The H1/H3 numbers above were measured with the
  system 2.6.0a1; across nodes 2.7.0 gives the same, H3 with two runs each
  in one job: makespan p50 0.48/0.49 s against 0.47/0.47 s at 2 nodes and
  1.72/1.82 s against 1.75/1.72 s at 64 nodes, recovery 0.45-0.46 s on both.
- Each sweep cost about 32-35 SUs (Stampede3 charges 15 minutes minimum per
  job; each job took 1-5 minutes).

Also on Stampede3:

- **Idle PEs spin at 100% of a core each** on Linux SKX (4 idle PEs: about
  407%). Reconverse branch `sleep-on-idle` (`4989e9f`, not merged, pin not
  moved) adds `+CmiSleepOnIdle`: 11-14% for 4 idle PEs, with kill detection,
  `replica_bench.py` and `contract.py` (59/59) unchanged. H1 at 4 nodes was the
  same with and without it (makespan 0.58-0.59 s); it matters where worker
  processes compete with PEs for cores.
- **Stalls that look like hangs** on a first launch were cold reads of the venv
  from `$WORK` (a PE blocked in `read()` inside `site`); staging the venv to
  `/tmp` removes them.

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
  `ray.cancel`, no `max_concurrency > 1` or async actors. Measured on
  Stampede3 (2026-10-19): a SIGKILLed node hangs the whole job over both
  `tcp` and `verbs;ofi_rxm`, because Reconverse's LCI2 backend
  (`comm_backend_lci2.cpp`, `issueAm`) retries a send to a dead peer forever
  and has no error path; a SIGSTOPped node leaves the others serving, with
  calls to it timing out. Surviving node loss needs that loop to give up and
  report the peer, then charm4py to mark its actors dead (as Ray does);
  checkpoint/restart does not fit (it restarts the whole job).
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

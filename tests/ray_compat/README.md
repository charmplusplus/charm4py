# Ray Core compatibility tests

`contract.py` is a single test program written against plain `import ray`. It
runs unchanged against real Ray (the oracle) and against `charm4py.ray`, so the
two implementations are checked against the same expectations.

Compatibility target: the Ray Core subset used by vLLM's `RayExecutorV2`
(vLLM `7fe8fc8`), following Ray **2.56.1**.

## Running

Oracle (real Ray, in its own virtual environment):

    python3.13 -m venv ~/venvs/ray_oracle
    ~/venvs/ray_oracle/bin/pip install "ray==2.56.*" numpy
    ~/venvs/ray_oracle/bin/python tests/ray_compat/contract.py

Charm4py (here on Reconverse; `LCRUN` points at LCI's launcher):

    export LCRUN=/path/to/lci/lcrun
    python -m charmrun.start +p4 -m charm4py.ray.launch tests/ray_compat/contract.py

`charm4py.ray.launch` installs `charm4py.ray` as the `ray` package and runs the
program with Ray's process model: the driver script runs in a process of its
own (served by PE 0), and tasks and actors run in worker processes owned by the
PEs (see "How it works" below). Every PE and worker imports the script as
`__mp_main__` (like multiprocessing's spawn mode), so driver code must be under
`if __name__ == '__main__':`. `--in-process` instead runs the driver, tasks and
actors as coroutines inside the Charm4py processes.

Arguments select tests by substring, e.g. `contract.py test_wait test_kill`. The
last line of output is a JSON summary.

The suite assumes one node with 4 CPUs and 2 GPUs, which it requests from
`ray.init(num_cpus=4, num_gpus=2)` (GPUs are logical; no GPU is needed).

## Status (2026-10-09, Apple M-series, Reconverse `reconverse-darwin-arm8`)

| Backend | Result |
|---|---|
| Ray 2.56.1 | 58/58 |
| charm4py.ray (launcher, worker processes), `+p1` `+p2` `+p4` `+p8` | 58/58 |
| charm4py.ray `--in-process`, `+p4` | fails the isolation tests, and driver threads and actor crashes take a PE down (by design) |

## How it works

Charm4py processes (one per PE) run the runtime: object store, task dispatch,
actor hosts, and the control plane. User code runs in worker processes that do
not load Charm++:

- the **driver** process, served by `_DriverHost` on PE 0;
- one **actor** process per actor, owned by the actor's `_ActorHost` element,
  which orders its calls and stores their results;
- a pool of **task** processes per PE, owned by `_RayWorker` (at most the node's
  CPUs divided among its PEs run at once; a task blocked in `ray.get`/`ray.wait`
  frees its slot, so nested tasks cannot deadlock the pool).

A worker's `ray` calls are forwarded to its host over a socket (any thread may
call); references cross that boundary as tokens (`_ipc.py`) so Charm4py's
borrower protocol never sees a non-Charm process. Values made by user code
(arguments, results, errors, `ray.put` objects) are serialized once, in the
worker that makes them; the Charm4py processes store and forward them as opaque
bytes and never import the user's modules (as Ray's raylet never deserializes
objects). Only top-level ObjectRef arguments are visible to the hosts, which
resolve them.

Objects are freed once no process references them (`_refs.py`): between PEs a
borrower tree with no message-order assumptions, between a PE and its workers
exact token counts. `refcount_stress.py` passes references nested in arguments,
results, `put` objects and actor state across PEs and drops them while copies
are in flight; every value arrives and the object store ends empty (`+p1` to
`+p8`). Hosts poll their workers'
sockets from a Charm timer, every 0.2 ms when active and every 10 ms when idle.
Isolation gives each actor its own `os.environ`, `CUDA_VISIBLE_DEVICES` (from its
assigned GPUs) and lifetime: `ray.kill` terminates the process, and a crash
becomes `ActorDiedError`.

Each `ray.init()` begins a Ray *job* (its own job ID, namespace and
runtime_env); the first one also starts the runtime, with its resources. Besides
the driver, any process started by the job's processes can call `ray.init()`:
the launcher puts a directory on `PYTHONPATH` whose `ray` package loads
charm4py.ray, and sets `CHARM4PY_RAY_ADDRESS`, through which the process
connects to `_Gateway` on PE 0 (`get_runtime_context().gcs_address` gives the
same address, for `RAY_ADDRESS`). This is how vLLM's engine process -- forked or
spawned by the driver, which never calls `ray.init()` -- uses Ray
(`engine_process.py` checks that layout on both backends). A job's actors and
placement groups end when it calls `ray.shutdown()` or its process exits; the
runtime stops with its last job. GPUs are detected per node as Ray does
(`CUDA_VISIBLE_DEVICES`, else NVML or `nvidia-smi`) unless `num_gpus` is given.

Programs that use `charm.start` directly default to in-process execution, since
their tasks and actors may call Charm4py; `ray.init(_isolation='process')` opts in.

### Control plane

There is no central scheduler (`_control.py`). Each PE coordinates the actors
and placement groups created on it (life cycle, kill, death, waiting for
resources); each node's last PE is the only authority for that node's
resources and accelerator IDs, and places actors on the node's PEs; names are
hashed over all PEs. Actor placement asks node controllers in turn, the local
node first; placement groups reserve bundles in two phases among only the nodes
they use. Waiting work is retried when a node it asked frees resources. Only the
job table stays on PE 0. `multinode.py` checks placement across nodes, the
placement group strategies and reservation under contention; with
`CHARM4PY_RAY_FAKE_NODES=k` it splits one machine's PEs into k nodes, and it
runs unchanged on Ray (on 3 raylets on one machine):

    CHARM4PY_RAY_FAKE_NODES=3 python -m charmrun.start +p6 -m charm4py.ray.launch \
        tests/ray_compat/multinode.py                                  # 8/8
    cd tests/ray_compat && python -c 'import multinode as m; m.ray_oracle_cluster()'

## vLLM

`vllm_generate.py` runs greedy generation through vLLM's `RayExecutorV2`
(unmodified vLLM `7fe8fc8`). On a machine without GPUs, the test-only platform
plugin in `vllm_plugin/` (`pip install -e vllm_plugin`, active with
`VLLM_CPU_RAY=1`) is vLLM's CPU platform with each worker's slot exposed to Ray
as a logical GPU, so the Ray executor runs on CPUs with gloo collectives:

    VLLM_CPU_RAY=1 VLLM_CPU_KVCACHE_SPACE=1 python -m charmrun.start +p4 \
        -m charm4py.ray.launch --num-gpus 2 tests/ray_compat/vllm_generate.py 2

`--num-gpus` plays the part of `ray start --num-gpus`: vLLM's engine process
calls `ray.init()` without resources. Results on the M-series Mac
(`facebook/opt-125m`, 4 prompts x 24 tokens, `VLLM_CPU_OMP_THREADS_BIND=nobind`,
two runs each); the text is identical across all of them:

| Executor, TP=2 | `LLM()` startup | generate |
|---|---|---|
| vLLM multiprocessing (no Ray) | 8.7 s, 8.8 s | 1.51 s, 1.52 s |
| Ray 2.56.1 (`ray start` beforehand) | 8.8 s, 8.9 s | 1.70 s, 1.71 s |
| charm4py.ray, `+p4` | 8.6 s, 8.5 s | 1.53 s, 1.53 s |

vLLM's own executor tests also run unmodified under the launcher (`-m` runs a
module as the driver, like `python -m`):

    cd vllm && VLLM_CPU_RAY=1 VLLM_CPU_KVCACHE_SPACE=1 python -m charmrun.start +p4 \
        -m charm4py.ray.launch --num-gpus 4 -m pytest tests/distributed/test_ray_v2_executor.py

21 of 23 pass (worker death and the failure callback, collective RPC, shutdown,
external placement groups, `VLLM_RAY_BUNDLE_INDICES`, TP=1/2/4 and PP=2
executors, generation at TP=2); the 2 failures are generation at PP=2, below.

TP=1 matches too. PP=2 fails the same way with vLLM's own multiprocessing
executor (`intermediate_tensors` is None on the second stage): a vLLM CPU
issue, not a Ray one.

## What charm4py.ray implements

- `ray.init` (`num_cpus`, `num_gpus`, `resources` describe each node and may
  only be given by the job that starts the runtime; `address`/`RAY_ADDRESS`
  connect to this job, other addresses are ignored), `is_initialized`,
  `shutdown` (ends this job: kills its actors and placement groups; `ray.init`
  can be called again). Jobs: separate job IDs and default namespaces, job
  `runtime_env` merged into its actors' and tasks', fate-sharing with the
  process that began the job.
- Remote functions: `@ray.remote`, `@ray.remote(**options)`, `.options()` (returns
  a copy), positional and keyword arguments, `num_returns` (0, 1, n). Top-level
  ObjectRef arguments are resolved; nested ones are passed as references.
- Actors: construction with keyword arguments, per-handle call ordering,
  `@ray.method(num_returns=...)`, `name` / `namespace` / `get_if_exists`,
  `ray.get_actor`, `ray.kill`, `ray.actor.exit_actor`, `__ray_ready__`,
  `__ray_terminate__`, handles passed to tasks and created inside tasks.
- `ray.get` (single ref or list, `timeout`), `ray.wait` (`num_returns`,
  `timeout`, input-ordered results), `ray.put`, Ray's argument validation errors.
- Errors: `RayTaskError` (also an instance of the original exception class),
  dependency failures, `GetTimeoutError`, `RayActorError`/`ActorDiedError`
  (constructor failure, kill, exit), `ActorAlreadyExistsError`.
- `ray.ObjectRef` (ID equality and hashing, `hex()`), `ray.types.ObjectRef`
  (annotation-only, as in Ray), `ray.actor.ActorHandle`.
- Topology: `ray.nodes()` (one node per physical host, not per PE),
  `cluster_resources`, `available_resources`,
  `ray._private.state.available_resources_per_node`,
  `ray.util.get_node_ip_address`.
- `get_runtime_context()`: node, job, task, actor and worker IDs, actor name,
  accelerator IDs, placement group ID, namespace, runtime_env.
- Placement groups: `PACK`, `SPREAD`, `STRICT_PACK`, `STRICT_SPREAD`, atomic
  reservation, `ready()`, `wait()`, `bundle_specs`, `placement_group_table`,
  `remove_placement_group` (kills the actors in it), `get_current_placement_group`,
  `PlacementGroupSchedulingStrategy`, `NodeAffinitySchedulingStrategy`, pending
  placement until resources are released. Accelerator IDs are assigned per bundle.
- Options that cannot be honored yet raise `NotImplementedError` when set to a
  non-default value (`max_concurrency > 1`, `max_restarts`, `concurrency_groups`,
  `lifetime='detached'`, `runtime_env` keys other than `env_vars`, label
  selectors, fractional accelerators, ...); unknown options raise Ray's
  `ValueError`.

## Known gaps

The main ones:

- Tasks are dispatched round-robin to PEs, without resource admission;
  task `num_gpus` gets no accelerator IDs and no `CUDA_VISIBLE_DEVICES`.
- No restarts or retries (`max_restarts`, `max_task_retries`,
  `retry_exceptions`), no detection of a dead node or Charm4py process.
- Objects move between worker processes by pickling over sockets (no
  shared-memory object store); no memory limits or spilling; no lineage
  reconstruction of lost objects.
- Not implemented: `ray.cancel`, `max_concurrency > 1`, async actors,
  concurrency groups, streaming generators, DAGs, `runtime_env` beyond
  `env_vars`, detached actors, connecting from a host other than PE 0's or from
  a process the job did not start (it lacks the `import ray` shim), `ray.util`
  beyond placement groups and scheduling strategies, the state API.

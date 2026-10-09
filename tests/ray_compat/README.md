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
| Ray 2.56.1 | 52/52 |
| charm4py.ray (launcher, worker processes), `+p1` `+p2` `+p4` `+p8` | 52/52 |
| charm4py.ray `--in-process`, `+p4` | fails the isolation tests (by design) |

## How it works

Charm4py processes (one per PE) run the runtime: object store, task dispatch,
actor hosts, and `_RayControl` (PE 0: names, resources, placement groups). User
code runs in worker processes that do not load Charm++:

- the **driver** process, served by `_DriverHost` on PE 0;
- one **actor** process per actor, owned by the actor's `_ActorHost` element,
  which orders its calls and stores their results;
- a pool of **task** processes per PE, owned by `_RayWorker` (at most the node's
  CPUs divided among its PEs run at once; a task blocked in `ray.get`/`ray.wait`
  frees its slot, so nested tasks cannot deadlock the pool).

A worker's `ray` calls are forwarded to its host over a socket (any thread may
call); references cross that boundary as tokens (`_ipc.py`) so Charm4py's
borrower protocol never sees a non-Charm process. Hosts poll their workers'
sockets from a Charm timer, every 0.2 ms when active and every 10 ms when idle.
Isolation gives each actor its own `os.environ`, `CUDA_VISIBLE_DEVICES` (from its
assigned GPUs) and lifetime: `ray.kill` terminates the process, and a crash
becomes `ActorDiedError`.

Programs that use `charm.start` directly default to in-process execution, since
their tasks and actors may call Charm4py; `ray.init(_isolation='process')` opts in.

## What charm4py.ray implements

- `ray.init` (`address` is ignored; `num_cpus`, `num_gpus`, `resources` describe
  each node), `is_initialized`, `shutdown` (kills this job's actors and
  placement groups; `ray.init` can be called again).
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

- The control plane is centralized: `_RayControl` (one chare on PE 0) decides
  actor placement and owns all resources, names and placement groups.
- Tasks are dispatched round-robin to PEs, without resource admission;
  task `num_gpus` gets no accelerator IDs and no `CUDA_VISIBLE_DEVICES`.
- No restarts or retries (`max_restarts`, `max_task_retries`,
  `retry_exceptions`), no detection of a dead node or Charm4py process.
- References are never freed; objects move between worker processes by
  pickling over sockets (no shared-memory object store).
- Not implemented: `ray.cancel`, `max_concurrency > 1`, async actors,
  concurrency groups, streaming generators, DAGs, `runtime_env` beyond
  `env_vars`, detached actors, joining a cluster by `address`, `ray.util` beyond
  placement groups and scheduling strategies, the state API.

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

`charm4py.ray.launch` installs `charm4py.ray` as the `ray` package, imports the
script on every PE as `__mp_main__` (like multiprocessing's spawn mode) and runs
it on PE 0 as `__main__`. Driver code must be under `if __name__ == '__main__':`.

Arguments select tests by substring, e.g. `contract.py test_wait test_kill`. The
last line of output is a JSON summary.

The suite assumes one node with 4 CPUs and 2 GPUs, which it requests from
`ray.init(num_cpus=4, num_gpus=2)` (GPUs are logical; no GPU is needed).

## Status (2026-10-09, Apple M-series, Reconverse `reconverse-darwin-arm8`)

| Backend | Result |
|---|---|
| Ray 2.56.1 | 47/47 |
| charm4py.ray, `+p4` / `+p8` | 47/47 |
| charm4py.ray, `+p2` | 46/47 (`test_wait_order_and_num_returns`) |
| charm4py.ray, `+p1` | 42/47 |

The `+p1`/`+p2` failures all come from the same limitation (below): tasks and
actors that block in `time.sleep` occupy their whole PE.

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

- **Process model.** Tasks and actors run as coroutines inside the Charm4py
  process of a PE. Code that blocks without yielding (`time.sleep`, a busy loop,
  a long native call) stops every other task, actor and runtime message on that
  PE. Consequences: concurrency is bounded by the number of PEs (PE 0, the
  driver's, is not used for tasks or actors when there are others); with `+p1`
  such code also blocks the driver; `ray.kill` cannot stop a running method (its
  pending results fail at once, but the code runs to completion); actors in one
  process share `os.environ`, so `runtime_env["env_vars"]` is process-wide.
  This is Phase 3 (actor isolation) of the plan.
- No resource admission for tasks: task `num_cpus`/`num_gpus` are validated but
  tasks are dispatched round-robin to PEs regardless.
- `ray.get`/`ray.wait` must be called from the driver, a task or an actor
  method -- not from other Python threads (vLLM's `RayWorkerMonitor` thread does
  this; also Phase 3).
- Not implemented: `ray.cancel`, DAGs (`.bind`), async actors, streaming
  generators, object spilling and lineage reconstruction, failure detection of
  crashed processes, autoscaling. `ray.init(address=...)` cannot join a
  separately started cluster.
- References are never freed (their futures stay registered), so long-running
  programs grow.

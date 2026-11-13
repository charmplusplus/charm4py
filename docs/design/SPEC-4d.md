# Step 4d specification: pool, Ray, interactive mode, CCS, liveViz, appspector

Written 2026-11-08 with the implementation (branch `thin`), against `ckdyn-dev`
c81fbeecc. It extends SPEC-4a.md, SPEC-4b.md and SPEC-4c.md. DESIGN.md 7.1 step 4d
is the gate; the logs are in `logs/gate-4d/`.

## 1. What each feature needed

| feature | state on `thin` before 4d | change |
|---|---|---|
| `charm.pool` (`map`, `map_async`, `Task`, `submit`, exceptions in tasks, `ncores`) | worked: `PoolScheduler` is the one-element array of DESIGN.md 5.4 item 13 (`Chare(PoolScheduler, onPE=0, migratable=True)` in `_createInternalChares`), workers an `Array(Worker, numPes)`; `self.workers.elemIdx = (i,)` goes through the Python `elemIdx` setter of `chare.py` | none |
| Ray tasks (`@ray.remote` functions, run by the pool) | arguments that are store futures reached the task unresolved: `add_task(e, f)` received two `Future` objects | `ProxyMethod` (binding): a reserved keyword (`ret`, `awaitable`, `is_ray`) also fills a missing positional parameter of the same name, as in 1.x (section 2) |
| Ray actors (`@ray.remote` classes) | `AttributeError: cid` in `RayProxy.__init__` | `ray/api.py`: wrap only the proxy's entry methods (attributes with an `ep`); `dir()` of a `_ckdyn.Proxy` lists `gid`, `cid`, `aid`, of which an element proxy raises `AttributeError` for the kinds it is not |
| object store | ported in 4a (`src/objstore_py.cpp`) | none |
| interactive mode and remote exec (`rexec`, `exec`, `eval`, `registerNewChareType(s)`, late registration) | worked | none |
| CCS (`CcsRegisterHandler` ... `CcsSendDelayedReply`) | the binding calls `CkDynCcs*`; a request crashes PE 0 (section 4) | none in charm4py; a reconverse defect, reported |
| liveViz, steer | unit tests pass; live requests need CCS (section 4) | none |
| appspector | `AttributeError: recvArrayMsg` at `serve()`: it wrapped the 1.x dispatch functions, which no longer exist, and the fast path calls the entry method from C++ | a load meter in the binding (section 3) |

## 2. Reserved keywords that are also parameters

`Pool.map_async(..., is_ray=True)` calls `pool_scheduler.start(func, tasks, result,
ncores, chunksize, is_ray=is_ray)`, and `PoolScheduler.start` has a parameter
`is_ray`. In 1.x the proxy method mapped every keyword onto missing positional
parameters by name before it popped `is_ray`, so `start` received `is_ray=True`
(and the message also carried the RAY flag with the appended store future, which
`_run` pops as the return future). The 4a `ProxyMethod` skipped the reserved names
when mapping, so `start` got its default `False`, the job was not marked as a Ray
job, and the workers' messages had no RAY flag, so `_invoke` never resolved the
store futures among the task's arguments. `pm_call` now maps any keyword,
reserved or not, onto a missing positional parameter of that name; a reserved
keyword keeps its send meaning as well. A user keyword that matches no parameter
still raises `TypeError`.

## 3. appspector: the load-data path

CkDyn has no call that reads an object's load out of the load balancer's database
(`CkDynSetObjLoad` writes it; nothing reads it), and appspector's numbers are not
the balancer's anyway: per PE, the wall time, the idle time, the exclusive time
of every chare, and the exclusive time of every entry method, measured with
`time.perf_counter()` around the calls that run Python code. So the binding gives
appspector the call boundaries instead of a database:

- `_ckdyn.set_meter(enter, leave)` (per PE; `set_meter(None, None)` stops). While
  it is set, the binding calls `fr = enter(obj, ep)` before and `leave(fr, obj)`
  after: every `deliver` (before argument conversion, so conversion is charged);
  every element of a `deliver_many` (the shared conversion in an outer frame with
  no chare, as 1.x's `recvArrayBcast`); every `construct` (`obj` is `None` at
  `enter` and the new object at `leave`); every `unpack` (no entry method). A
  section multicast arrives as one `deliver` per member, so members are charged
  separately with no section-manager special case. When no meter is set the cost
  is one test of a thread-local pointer per hook call.
- Timers: while a meter is set, `call_after` looks up `charm.triggerCallable` at
  each call instead of using the reference taken at `init`, so appspector's
  wrapper (an instance attribute) times them and keeps its own gather timer out.
- Coroutines: resumes go through `threadMgr.resumeThread` and pauses through
  `threadMgr.pauseThread`, Python attributes as in 1.x; appspector wraps them as
  before (time after a resume is charged to the chare and entry method the
  coroutine started in).
- `Tracker.run(key, ep, fn, args)` became `enter`/`leave` (same stack and
  exclusive-time rules, same handling of frames left open by a paused coroutine);
  `Tracker.install(charm, threadMgr, set_meter)` installs the meter. The
  `fanout_gids` parameter is gone (no section manager group on `thin`).
  `tests/appspector/state.py` replaces its section-fanout case with a broadcast
  case (outer conversion frame, per-element frames, an inline call nested in one
  element, a construction charged to the object it returns).

Not measured, as in 1.x: the main chare's constructor; time in C++ outside the
hooks (counted as idle). Future deposits from another PE are charged only through
the coroutine they resume (1.x charged them to the `CharmRemote` group).

## 4. CCS: a reconverse defect blocks live requests

In the `charm4py` build of reconverse (`CMK_CHARM4PY` defined),
`callHandlerRec` in `contrib/reconverse/src/ccs/conv-ccs.cpp` (lines 36-57) always
calls `c->fnOld` with the Ext-API message layout. `CkDynCcsRegister` registers its
trampoline with `CcsRegisterHandlerFn`, which sets `fn` and `userPtr` and leaves
`fnOld` NULL, so the first request to any handler registered by charm4py jumps to
address 0: `steer_demo.py ++server ++server-port 12731`, then any request to
`c4pSteer`, `lvConfig` or `lvImage` ends PE 0 with
`Fatal signal 11 (SIGSEGV)` in `CcsHandleRequest` ← `req_fw_handler`
(`logs/gate-4d/steer_demo_p1.log`). The fix is in reconverse: in the `CMK_CHARM4PY`
branch of `callHandlerRec`, call `c->fn(c->userPtr, reqLen, reqData)` when
`c->fnOld == NULL` (or drop the branch once the Ext API goes, DESIGN.md 3.15,
which already lists this as a reconverse pull request). Registration and startup
work (`++server` opens the port; `examples/liveviz/liveviz.py` registers its
handlers and waits). `logs/gate-4d/ccs_client.py` is the check to run once the fix is in:
it calls `list`, `set_waves`, `set_color`, `state`, `slow`, an invalid call and
`eval` (refused without `++server-auth`) on `steer_demo.py`, then `lvConfig` and
three `lvImage` requests, and prints `CCS CHECK PASS` or `FAIL`.

## 5. Skipped, and why

- The numba variants (`examples/fibonacci/fib-numba.py`, `examples/nqueen/nqueen-numba.py`,
  `examples/wave2d/wave2d.py`): numba is not installed in the venv; `test_config.json`
  makes them conditional on it.
- One-process runs of the examples that use `charm.pool` (`parallel-map/square.py`,
  `ray/simple.py`, `ray/fib.py`, `ray/mandelbrot/mandelbrot.py`): the pool needs a
  worker PE besides PE 0 (`AssertionError: Run with more than 1 PE to use
  charm.pool`, pool.py:105, unchanged from 1.x). auto_test runs them at 4.
- Live CCS requests (steer, liveViz, appspector's handlers): section 4.

`torch`, `torchvision` and `matplotlib` were installed in the venv for
`ray/parameter_server.py` and `ray/mandelbrot/mandelbrot.py`.

## 6. Gate results (logs/gate-4d, 2026-11-08)

Runner: `logs/gate-4d/gate4d.sh` (one process: `python X +pe 1`; N processes:
`lcrun -n N python X +pe N`; counts are 1 and 2, or `force_min_processes`);
`run_interactive.sh` pipes `tests/interactive/test1.in` to `charm4py.interactive`
through `charmrun.start` as auto_test does. Summary in `final-pass.txt`.

| test | processes | result |
|---|---|---|
| tests/pool/pool.py, pool_ncores.py, tests/exceptions/pool.py | 4 | pass |
| tests/appspector/loads.py | 2 (also 4 in the 4b rerun) | pass |
| tests/interactive/test1.in | 4 | pass |
| examples/ray/simple.py, fib.py, mandelbrot/mandelbrot.py 100 100 10 10 | 2 / 1 | pass / pool refuses one PE |
| examples/ray/parameter_server.py sync, async | 1, 2 | pass |
| examples/parallel-map/square.py | 2 / 1 | pass / pool refuses one PE |
| examples/dist-task-scheduler, fibonacci/fib, hello/group_hello, hello/array_hello, multi-module, nqueen | 4 | pass |
| examples/jacobi/jacobi2d.py 128 32, examples/simple/{start,chares,reduction,hello_world}.py | 1, 2 | pass |
| tests/exec.sh examples/simple/hello_world.py | 2 | pass |
| tests/liveviz/test_liveviz_unit.py (37), tests/steer/test_steer_unit.py (25) | unittest | pass |
| examples/steer/steer_demo.py with a CCS client | 1 | PE 0 SIGSEGV on the first request (section 4) |
| examples/liveviz/liveviz.py ++server, no request | 1 | starts, registers its handlers |
| numba examples | - | skipped, numba absent |

Regression: the 4a gate (`gate4a.py 1 2 4`, 150 runs) all pass; the 4b gate the
same as at 4c except `appspector/loads`, which now passes at 2 and 4 processes
(the one-process runs of slice, callbacks, constrained_groups, allreduce and loads
abort as before: they address PEs that a one-PE job does not have); the 4c gate
passes at 2 and 4 processes.

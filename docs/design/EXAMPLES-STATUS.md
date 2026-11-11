# Status of the 1.x examples on the thin charm4py

Owner: Ben (Zonghang) Wu, per `TASK-examples.md`. Last updated 2026-11-10 (batch_prediction
row, behaviour note 1, GPU examples). Started 2026-11-08 by a first sweep
on Kale's Mac (branch `thin` at 9001e96, reconverse-darwin-arm8, one PE per process,
each configuration run twice). Ben completes the 1.x column, re-runs on Linux, judges
the "differs" cases and files the `[thin]` issues.

## Covered by test_config.json

The 20 examples listed in `test_config.json` pass in CI on Ubuntu and macOS at 1 and 2
processes (and 4 where the entry asks for it) and on NCSA Delta (two A40 GPUs, CUDA
13.2), except the four `charm.pool` examples at one process, which stop with "Run with
more than 1 PE to use charm.pool" on 1.x as well. `examples/cuda/hapi/*` pass on Delta.

## Not covered by test_config.json (first sweep)

| example | args | +p1 | +p2 | +p4 | verdict |
|---|---|---|---|---|---|
| cannon/cannon.py | 64 4 | PASS x2 | PASS x2 | PASS x2 | runs unchanged. A copy with a per-block result check (variants/cannon_check.py) is correct at p1/p2/p4, at 512 8, and at 1536 3 (2 MiB blocks through channels). |
| fibonacci/fibonacci_with_futures.py | 25 | PASS x2 | PASS x2 | PASS x2 | 75025, correct |
| hwmon/hwmon.py | none | FAIL x2 | FAIL x2 | - | environment: `FileNotFoundError: 'sensors'` (lm-sensors is Linux only), hwmon.py:58. With line 58 replaced by a fixed "Core 0: +60.0C" line (variants/hwmon_fake.py) PASS x2 at p1 and p2: ret=True group call, scheduleCallableAfter, charm.sleep, getHostFirstPe all work, exits after 30 s. |
| miniapps/LeanMD/main.py | - | NOT RUN | NOT RUN | NOT RUN | needs numba (`import numba` at line 3; `@numba.njit` on methods of class Physics relies on the dispatcher not binding `self`, so a no-op decorator is not a two-line fix) |
| miniapps/MiniWeather/miniweather.py | - | NOT RUN | NOT RUN | NOT RUN | needs numba.cuda and an NVIDIA GPU (`from numba import cuda`, line 6) |
| mnist/mnist.py | none (6 epochs) | FAIL x2 | PASS x2 (36 s, 34 s) | PASS x2 (29 s, 30 s) | p1 failure is the example, same on 1.x (see finding 3). p2/p4 print torch "NumPy array is not writable" warning: finding 1. Loss falls 1.31 -> 0.29. |
| mnist/mnist-mpi4py.py | - | NOT RUN | NOT RUN | NOT RUN | needs mpi4py |
| multi-module/hello.py | none | exit 0 | exit 0 | - | hello.py is a module without `charm.start`; it prints "charm was not started" and exits 0. hello.py does not import goodbye.py; main.py imports both. |
| multi-module/main.py | none | PASS x2 | PASS x2 | PASS x2 | runs unchanged (it is in test_config.json too) |
| pool/pool_fibonacci.py | none | expected stop x2 | PASS x2 | PASS x2 | fibonacci(13)=233 |
| pool/pool_simple.py | none | expected stop x2 | FAIL x2 | FAIL x2 | example bug, same on 1.x: `TypeError: unsupported operand type(s) for *: 'int' and 'Future'` at pool_simple.py:8. Line 13 passes the list of futures `results` as task inputs; the pool does not resolve futures. With `[r.get() for r in results]` on line 13 (variants/pool_simple_fix.py) PASS x2 at p2 and p4, prints 32. |
| ray/batch_prediction.py | none (reads 12 parquet shards from S3) | expected stop x2 | FAIL x2 | FAIL x2 | Run 2026-11-08 17:51, before the fix: `KeyError: (3, 1)` in `notify_future_deletion` (charm.py:1291), finding 4, a `ray.put` reference shared by several tasks. FIXED since on thin (7ff1085, 2026-11-08 22:15) and on 1.x (PR #279, merged to main 2026-11-09). **Rerun needed on both branches.** With `model` instead of `model_ref` on line 51 (variants/batch_prediction_noref.py): all 12 "Prediction output size" lines print (p2, p4), then exit code 139: pyarrow "FinalizeS3 was not called", SIGSEGV/"mutex lock failed" in the S3 event-loop thread at process exit, finding 5 (still open; check whether 1.x does the same). |
| ray/model_selection.py | 4 and 8 | expected stop x2 | PASS x2 | PASS x2 | best accuracy varies run to run (random hyperparameters), expected |
| sssp/sssp.py | 1000 5000 1 0 and 1000 20000 1 0 | PASS x2 | PASS x2 | PASS x2 | 5000 edges: max cost inf (some vertices unreachable at mean degree 5, plausible). 20000 edges: 1.088062737724951 at every process count, equal to a sequential Dijkstra on the same generated graph. |

"expected stop" = `AssertionError: Run with more than 1 PE to use charm.pool`, exit 134.
No HANG anywhere; no run needed a kill; no orphan processes. No difference between
run a and run b of any configuration except timings and model_selection accuracies.

"Expected stop" is the `charm.pool` assertion at one process, same as 1.x.

## 1.x column

To be filled by Ben: the same runs with charm4py 1.x (`pip install charm4py` or branch
`main`) in a second virtualenv. Already known from the sweep: `pool_simple.py`,
`mnist.py` at one process and `ray/batch_prediction.py` fail on 1.x the same way
(example bugs or a 1.x `ray.put` bug, see `logs/step-7/examples-sweep.md` findings 3-5
on Kale's laptop, summarized in the issues once filed).

## Behaviour differences found so far

1. The result of `allreduce` is a read-only numpy view on thin and writable on 1.x
   (`mnist.py:184`, torch warns). This is a documented 2.0 behaviour, not a defect:
   `docs/migrating.rst` ("Received numpy arrays are views, and some are read-only") and
   DESIGN.md 5.4 item 8. `charm.options.copy_received_arrays = True` restores the 1.x
   behaviour. No issue to file; an example that writes to the result should copy it.
2. (2026-11-10) `examples/cuda/` on thin now has, from PR #319, the Jacobi3D, bandwidth and
   multi-array GPU examples ported from the Charm++ tree, with results on NCSA Delta
   (A100 pair) in the PR description. They need a CUDA build and cupy or numba; not part
   of this task unless you have a GPU node.
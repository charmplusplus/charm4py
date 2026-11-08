# Task: run the 1.x examples on the thin charm4py (branch `thin`)

Owner: Ben (Zonghang) Wu (GitHub login to be added). Mentor: Kale. Written 2026-11-08.

## Goal

Run every program under `examples/` on the thin charm4py and record, for each, whether
it runs unchanged, fails, or behaves differently from charm4py 1.x. The result is a table
in `docs/design/EXAMPLES-STATUS.md` and one GitHub issue per failure, so the failures feed
steps 4c and 4d of `docs/design/DESIGN.md`. No fixes to the package are expected from this
task; small fixes to an example are fine when the cause is clear (say so in the table).

## Setup (Linux or macOS, Python 3.11 or newer)

1. Clone Charm++ and check out branch `ckdyn-dev`:
   `git clone -b ckdyn-dev https://github.com/charmplusplus/charm && cd charm && git submodule update --init contrib/reconverse`
2. Clone charm4py branch `thin` next to it: `git clone -b thin https://github.com/charmplusplus/charm4py`
3. Build and install, following `charm4py/docs/install.rst` on branch `thin` (the
   reconverse build triplet for your machine; on a Mac it is `reconverse-darwin-arm8`,
   on Linux x86 `reconverse-linux-x86_64`). The install step builds libcharm with
   `-DREFNUM_TYPE="unsigned int"` and the `_ckdyn` extension with scikit-build-core.
4. Check: `python -m charmrun.start tests/thin/smoke_ckdyn.py +p1` prints PASS lines.

## Rules of the runtime you are testing

- One PE per process. `+pe 2` in one process is refused with a message. Multi-process
  runs use lcrun from the build directory: `<build>/_deps/lci-src/lcrun -n 2 python
  example.py +pe 2`, or `python -m charmrun.start example.py +p2`.
- Sections do not survive `+restart`; `charm.combine` accepts only array sections of one
  class and dimension count (DESIGN 5.4 item 15).
- Never run two lcrun jobs at once on one machine.

## What to record for each example

| example | 1 PE | 2 processes | 1.x result | notes |

Columns: pass, fail (with the last 10 lines of output and the exception), or
"differs" (what differs). Run each example twice at each process count; note any
nondeterministic failure. For the 1.x column run the same example with charm4py 1.x
installed in a second virtualenv (`pip install charm4py` or the `main` branch).

## Order

Start with examples that use only arrays, groups, futures, channels and reductions:
hello, simple, fibonacci, nqueen, jacobi, wave2d, dist-task-scheduler. Then the pool
examples (parallel-map, ray/fib), then particle (load balancing), then the Ray examples.
The last three groups depend on steps 4c and 4d, which are in progress; failures there
are expected and still worth recording precisely.

## Filing

One issue per distinct failure against charm4py, title starting with `[thin]`, body:
example, command, process count, output tail, and whether 1.x passes. Link the issues
from the status table. Group duplicates (same traceback) into one issue.

## Done when

Every example has a row with both process counts filled, and each failure has an issue.

# Design documents for Charm4py 2.0 (developed on branch `thin`)

| file | what |
|---|---|
| `DESIGN.md` | the design: layers, the CkDyn C API, the Python binding, the plan with gates, decisions and their dates |
| `SPEC-4a.md` | the contract for step 4a (the `_ckdyn` binding and the Python cut-over) |
| `SPEC-4b.md` | the contract for step 4b (sections in Python) |
| `SPEC-4c.md` | step 4c: migration, load balancing and checkpoint |
| `SPEC-4d.md` | step 4d: pool, Ray, interactive mode, CCS, liveViz, appspector |
| `SPEC-3.md` | zero-copy host buffers, device buffers, HAPI callbacks |
| `SPEC-6.md` | step 6: packaging |
| `SPEC-7-api.md` | NodeGroup, entry-method attributes, message priorities, trace names |
| `SPEC-7-coro.md` | migratable coroutine entry methods |
| `SPEC-7-device.md` | persistent device buffers, posted receives, IPC path |

Process documents are in `history/`:

| file | what |
|---|---|
| `history/ANALYSIS-2026-11-06.md` | the measurements and analysis that motivated the redesign |
| `history/PROMPT.md` | the task statement given to the design session |
| `history/REVIEW-2026-11-07.md` | the independent design review (findings F1-F20) and how each was resolved |
| `history/TASK-examples.md` | task contract: run the 1.x examples on `thin` (Ben Wu) |
| `history/EXAMPLES-STATUS.md` | the status of the 1.x examples on the new binding |

The C layer, CkDyn, lives in the Charm++ repository on branch `recharm` (merged from `ckdyn` by charm PR #4193 on 2026-11-12)
(`src/ck-core/ckdyn.h`, `doc/internals/ckdyn.md`). Paths of the form
`~/software/...` in these documents refer to the development laptop.

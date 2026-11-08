# Design documents for the thin charm4py (branch `thin`)

| file | what |
|---|---|
| `ANALYSIS-2026-11-06.md` | the measurements and analysis that motivated the redesign |
| `PROMPT.md` | the task statement given to the design session |
| `DESIGN.md` | the design: layers, the CkDyn C API, the Python binding, the plan with gates, decisions and their dates |
| `SPEC-4a.md` | the contract for step 4a (the `_ckdyn` binding and the Python cut-over) |
| `REVIEW-2026-11-07.md` | the independent design review (findings F1-F20) and how each was resolved |
| `SPEC-4b.md` | the contract for step 4b (sections in Python) |
| `TASK-examples.md` | task contract: run the 1.x examples on `thin` (Ben Wu) |

The C layer, CkDyn, lives in the Charm++ repository on branch `ckdyn-dev`
(`src/ck-core/ckdyn.h`, `doc/internals/ckdyn.md`). Paths of the form
`~/software/...` in these documents refer to the development laptop.

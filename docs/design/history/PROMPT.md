# Initial prompt: thin-shim redesign of charm4py

Read `CLAUDE.md` and `ANALYSIS-2026-11-06.md` in this directory first. The analysis
answers "is the difference significant?" (yes, in feature coverage, duplicated
semantics and the cost of adding features; not on the per-message hot path) and
sketches a three-layer design. Do not redo that analysis; build on it.

## Task

Produce a detailed design, pretty much from scratch, for a charm4py implementation
that is a thin shim over the complete Charm++ runtime, borrowing pieces from the
existing charm4py where they fit. Write it as `DESIGN.md` in this directory (it will
move into a repository later; keep it self-contained).

The design must cover, with concrete C++ and Python interfaces rather than prose
summaries:

1. Layer 2, the language-neutral dynamic-chare layer in C++ inside the Charm++ repo
   (`charm/` here, branch `ckdyn` off recharm). For each of: mainchare, chare, group,
   nodegroup, N-dimensional array element, array map, section manager over
   CkMulticastMgr: the generic C++ class, how it is registered with
   CkRegisterChare/CkRegisterEp at startup from a per-class entry table supplied by the
   binding, how entry method attributes (threaded, inline, expedited, immediate,
   nokeep, priority) are honored, and the message format (header plus typed slots:
   scalars, buffer-protocol zero-copy via CkNcpyBuffer, device buffers via
   CkDeviceBuffer, opaque serialized tail). Also: callbacks into dynamic objects,
   reductions (built-in types in C++, foreign reducer hook), pup through a function
   pointer serving both migration and checkpoint/restart, QD, load-balancing hooks
   including GPU attribution (CkCallstackPush/Pop), and tracing that shows the
   binding's method names in Projections. State what in the existing Ext API
   (src/ck-core/charm.h, the *Ext classes in charm++.h, ckarray.h, cklocation.h,
   register.C, ck.C) is kept, generalized, or deleted.
2. Layer 3a, the Python binding (nanobind, not Cython): dispatch table, argument
   conversion to slots and back, GIL and exception handling, greenlet integration for
   coroutines without a greenlet per invocation, numpy and cupy/torch zero-copy
   through `__array_interface__` and `__cuda_array_interface__`.
3. Layer 3b, the Python API layer: which charm4py modules are borrowed as-is, which
   are rewritten, which are deleted (expected: sections.py deleted, charm.py receive
   and creation paths deleted, reduction.py conversion tables deleted, Cython module
   deleted). The existing user-facing API and `tests/` are the acceptance suite;
   list any API changes explicitly and justify each.
4. Build and packaging: how the charm4py CMake target in the Charm++ tree changes,
   reconverse first (classic second only if cheap), wheels, CUDA/HIP builds.
5. Migration plan: ordered steps, each leaving a working system, with the tests that
   gate each step. Include the per-message and coroutine cost targets (today: 2.2 us
   and 6.0 us per message on one PE on this Mac, from the analysis) and how they are
   measured.
6. Risks and open questions, in particular ownership of Layer 2 inside recharm and
   what happens to the Ext API users other than charm4py (check whether any exist).

Work method: read the Charm++ sources named above and the relevant charm4py modules
before writing each section; cite file and line. Where a decision could go either
way, make a recommendation and say why. Mechanical surveys (listing every Ext API
call site, inventorying charm4py tests by feature) go to Opus agents.

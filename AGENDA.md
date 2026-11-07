# Charm4py project agenda

Drafted 2026-11-06 from the repository state (main at e97b12f), the open
PRs and issues, the Ritvik and Aditya 1-on-1 notes, and the appSpector
repository. Untracked file; move items to GitHub issues or the 1-on-1 docs
as they are assigned.

## 1. Merge queue (this week)

Four PRs by Advait, all CI-green on classic and Reconverse. Suggested order:

1. #313 GCC 14 build fix (Ritvik rewrote it, Advait verified on Delta).
   Follow-up in the charm repo: declare `LBTurnInstrumentOn/Off` and
   `CkExtContributeTo{Chare,Group,Array,Section}` in `charm.h` so charm4py
   can drop its private prototypes.
2. #310 liveViz repair (Ritvik's seven points addressed in 6da4d62).
   Follow-up issue: carry the request in the deposit (C++ liveViz's
   `hdr->req`) instead of matching replies to requests by queue position.
3. #311 appspector load data. Needs a docs page (`docs/*.rst`) before or
   right after merge; decide the module name (tool-neutral vs appspector).
4. #312 steer. Decide whether `eval` ships at all, or only behind
   `allow_eval` as proposed. Add the unit test to CI.

Cross-cutting: adopt one test convention for the three new modules.
Proposal: unittest files `tests/*/test_*_unit.py` run by one CI step
(`python -m unittest discover`), runtime programs via `test_config.json`.

## 2. Stale PRs and issues (housekeeping, one sitting)

- 11 open PRs from 2021-2024 (#208 GPU-aware communication, 2308 lines,
  draft; #235 concurrent.futures Executor; #221 nightly benchmarks; #224
  ckDestroy/ckLocalBranch; #223 cProfile; #230, #279, #216, #217, #212,
  #213). Each: close with a note, or assign a reviewer and a deadline.
  #279 (futures bug + pool PE check, Ritvik, 2024) looks mergeable.
- 25 open issues, most from 2018-2021. Triage into: still true on
  main (e.g. #266 import from project root, #263/#184 pool needs >1 PE,
  #238 ckLocal equivalent), feature wishes (#36 Jupyter, #19 MPI interop,
  #177 C++ interop), and obsolete (#219 Catalina, #179 Windows wheels).
- #270 meson-python migration: decide yes/no; setup.py currently derives
  the version from `git describe`, which breaks sdists.

## 3. Release 1.2

v1.1 (PyPI, 2025) predates: Reconverse support (#307, #308), CCS (#289),
liveViz (#298, #310), Projections support (#290), HAPI callbacks (#282,
#303), Python 3.14 fix (#306), mini-apps (LeanMD, MiniWeather), Ray
mandelbrot. Items:

- `build_wheels.yml` still pins Python 3.8 and macos-13; align with the
  CI matrix (3.10/3.12/3.14, macos-14/15/26, ubuntu).
- Decide which runtime the wheels embed: classic (today) or Reconverse,
  or both triplets. Reconverse wheels need LCI and libfabric bundled.
- CHANGES.rst entry, tag, PyPI upload; docs rebuild (readthedocs?).
- Default runtime in docs/install.rst: when does Reconverse become the
  default for charm4py users?

## 4. Live tools line (Advait)

- appSpector: viewer + ccsio (pure-Python CCS client). Open items from
  its ONBOARDING: per-PE view refinements, side-by-side layout,
  Reconverse `++server-auth` hex-key fix, Frontier/cloud transport.
- Where should ccsio live long term: appSpector, charm4py, or replace
  charmplusplus/PyCCS?
- Steering semantics beyond PE 0 functions: broadcast helpers, parameter
  registry, checkpoint trigger.
- Projections (#290) and appspector loads: one trace source or two?

## 5. Runtime and performance line (Ritvik, Aditya)

- Ray API on Charm4py: Ritvik's measurements show faster actor start and
  failure recovery than Ray; needs GPU scaling runs and shrink-expand.
  Decide whether this is a paper target (SC27) or maintenance only.
- Issue #309 GPU direct (device-to-device) messaging over LCI; Aditya's
  "portable d2d API" and charmnumeric slicing depend on it. Relation to
  the stale #208 GPU-aware communication PR.
- Async LB and the Channel API results (Aditya, Oct 29 notes): which
  parts land in charm4py vs charmnumeric/charmtyles?
- Local message optimization (#215, #225, #226): still disabled since
  v1.1; decide whether to fix or remove the code.
- Reconverse-specific gaps: AMPI-style migration across processes on
  macOS is blocked by ASLR (not charm4py-specific, but affects tests).

## 5a. GPU support: what Charm++ (recharm) now has and what charm4py can use

Surveyed 2026-11-06 against recharm `a49a5ef0c`. charm4py today exposes one
GPU call, `charm.hapiAddCudaCallback(stream, future)` (a HAPI stream
callback that sets a future), and needs libcharm built with the `cuda`
build option. Kernels, arrays and streams come from torch, numba or CuPy.

**GPU-resident data per chare.** Works now, with no runtime support
needed: a chare keeps torch/CuPy/numba device arrays as attributes, they
stay on the device across entry methods, and a kernel per entry method
plus the HAPI future gives asynchronous completion. Two gaps: (a) a device
array passed as an entry-method argument is pickled, so it crosses the
host; (b) migration pickles the chare, so device state also crosses the
host and the `+balancer` decision does not know its device footprint.
Charm++ itself now has a device region in the migration stream
(`PUP::DEVICE` buffers, 256-byte aligned, `setGPUPupSize`), but charm4py's
pickle-based pup never uses it. Aditya's `gpu-pup` branch (2025-07, 15
files) predates this and is not merged.

**Device-to-device messaging.** Merged on recharm and documented in the
manual ("Direct GPU Messaging"): `nocopydevice` parameters in the .ci,
`CkDeviceBuffer(ptr, size, cb, stream)` on the sender, a post entry method
on the receiver that supplies the destination device pointer
(`CkDeviceBufferPost`), `CkDevicePersistent` get/put for repeated
exchanges, `CkDeviceMalloc` for registered pool memory; on reconverse the
transfer is an LCI device RDMA get, also across nodes (reconverse #315,
pin bumped 2026-11-03). None of it is reachable from charm4py: charm4py
has no .ci and its Ext send/recv functions carry only host byte buffers
plus the host "dcopy" section. Reaching it needs new Ext entry points in
charm (send with device pointers and sizes, a receive callback that lets
Python post destination buffers, then the regular delivery) and the
charm4py side (a `Channel.send/recv` or entry-method argument form that
accepts device arrays via `__cuda_array_interface__`). Zane's 2021 PR
charm4py#208 plus the charm branch `charm4py_gpudirect` (Ritvik merged
main into it 2025-07; `CkArrayExtSendWithDeviceData`, `CkGetGPUDirectData`,
`registerArrayMsgGPUDirectRecvExtCallback`) did exactly this for the old
UCX path, so the Python API design exists; the C++ side has to be redone
on the new `CkDeviceBuffer`/`CkRdmaDeviceIssueRgets` code. That is issue
charm4py#309. Open design choice: expose it only through channels (as #208
did), or also as `nocopydevice`-style entry-method arguments.

**GPU-aware load balancing.** `GreedyRefineCentralGPULB` (Aditya, charm
#3968, merged 2026-09-13) is on recharm: CUPTI measures kernel time per
object by stamping every entry method via `CkCallstackPush/Pop`, the
balancer places objects across GPU groups by GPU time and within a group by
CPU time. CUPTI is detected automatically in a `cuda` build
(`HAPI_CUPTI_LB`). charm4py array elements are ordinary `CkMigratable`
objects (`ArrayElemExt`) whose Ext entry methods go through `ckInvokeEntry`
and the callstack, so kernels launched from Python inside an entry method
are attributed to the right chare with no charm4py change; `+balancer
GreedyRefineCentralGPULB` on the command line plus `useAtSync=True` and
`self.AtSync()` should just work, and libcharm links the LB modules.
Untested: a kernel launched after a coroutine resumes (the resume runs
inside the resuming message's callstack, so it may be charged to the wrong
object), and the device-footprint input (`gpuPupSize`), which charm4py
never sets. Aditya's `gpu-shrinkexpand` branch (2025-08, 68 files, 91
commits behind) is separate and not merged.

**Build and test gaps.** `reconverse-linux-x86_64 cuda` builds (charm
#3998), but charm4py's CI has no CUDA job and the PyPI wheels have no CUDA.
HIP: HAPI is portable (`hapi_portable.h`), charm4py's Cython layer only
checks a `HAVE_CUDA_BUILD` flag, so a HIP build is untested.

**Proposed order of work.**
1. Verify GPU LB from charm4py on Delta (one process per GPU, jacobi-style
   example with numba kernels, `+balancer GreedyRefineCentralGPULB`); add
   the example and a docs paragraph. Small, mostly a test.
2. Issue #309: device-to-device Ext API on recharm, Python side via
   `__cuda_array_interface__`; port the #208 jacobi3d example. Decide the
   owner (Advait? Aditya?) and whether channels-only is the first cut.
3. Device-aware migration: have charm4py's pup put device arrays in the
   device region instead of pickling them, and set `gpuPupSize`.
4. A CUDA CI job (self-hosted or Delta cron) and a `cuda` wheel variant,
   or document source builds as the only path.

## 6. Docs, tutorial, outreach

- Fold the IPDPS 2026 tutorial materials (charm4py-tutorial2026) into
  docs/tutorial.rst or link them; the Docker container and Codespaces
  setup from the tutorial are the easiest onboarding path.
- Docs pages missing: appspector/load monitoring, steer (in #312), CCS
  server API (#289 added code, is it documented?), Reconverse run idioms
  (`+pe`, `lcrun`) vs charmrun.
- Next tutorial or paper venue for charm4py itself (JLESC 2026 talk
  exists; is there a journal/software paper to cite besides the 2019
  one?). Zenodo DOI (#172) still open.

## 7. People and ownership

- Reviewer of record per area: core runtime/Cython (Ritvik), build and
  wheels (Ritvik), live tools/CCS (Advait), GPU/numerics (Aditya), docs
  (whoever changes the feature).
- Review turnaround target for external PRs, so the 2021-2024 backlog
  does not recur.

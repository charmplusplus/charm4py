# charm4py as a thin shim over Charm++: design

Status: first draft, written 2026-11-06/07 in the charm4py-thin project (Kale + Claude
Fable 5.1). Builds on `ANALYSIS-2026-11-06.md`; does not repeat its measurements.
Self-contained: it will move into a repository later.

Source baselines for every `file:line` citation:

| tree | commit | path on this machine |
|---|---|---|
| Charm++ (branch `ckdyn`, = `origin/recharm`) | `2a2d1ac67` | `~/software/charm4py-thin/charm` |
| charm4py (`main`) | `c5b288f` | `~/software/charm4py-thin/charm4py` |

Mechanical inventories that this document cites live next to it in `surveys/`:
`survey-ext-api.md` (every Ext API function and callback, with call sites),
`survey-tests.md` (every test and example by feature, with the reverse index used for
the gates in section 7), `survey-build.md` (how libcharm and the wheel are built today),
and `survey-charm-apis.md` (the Charm++ runtime APIs Layer 2 builds on).

---

## 0. Summary and decisions

The new charm4py has three layers. Layer 1 is the Charm++ core, unchanged except for a
handful of small, general additions listed in 3.15. Layer 2, **CkDyn**, is a
language-neutral C++ library inside the Charm++ repository (`src/ck-core/ckdyn.*`) that
makes it possible to define chare classes at run time from a table, and that owns every
piece of Charm++ semantics charm4py reimplements today: message format, dispatch to
objects, sections and multicast, reductions including foreign (binding-defined)
reducers, callbacks into dynamic objects, pup for migration and checkpoint, quiescence,
load-balancing and GPU attribution hooks, and tracing with the binding's method names.
Layer 3a is a nanobind extension (`charm4py._ckdyn`) that turns Python method calls into
CkDyn messages and CkDyn deliveries into Python method calls. Layer 3b is the existing
charm4py Python API with its receive, creation, section and reduction-conversion code
removed.

Decisions made in this document, each argued where it appears:

1. **CkDyn replaces the Ext API entirely.** No function from the current
   `CMK_CHARM4PY` blocks survives in `recharm`; the survey found no consumer of the Ext
   API other than charm4py (surveys/survey-ext-api.md section 4), and the frozen
   `classic` branch keeps the old API for the old charm4py. (3.15, 8.2)
2. **One payload format, `CkDynMsg`, for every dynamic entry method**, a varsize Charm++
   message whose body is a header plus typed slots, with a second, layout-different
   type for section multicasts because CkMulticast writes its cookie into the first
   bytes of a message and the zero-copy code reads descriptors from there. Scalars and small blobs are inline, host
   buffers are copied or carried by `CkNcpyBuffer`, device buffers by `CkDeviceBuffer`,
   and an opaque tail holds whatever the binding serializes itself. The current
   `(header dict, args list)` pickle disappears from the common path. (3.4)
3. **The binding's object is reached by pointer, not by lookup.** Each CkDyn object
   stores an opaque `void *handle` returned by the binding's construct hook, and every
   delivery passes it back. The dictionary lookups `charm.arrays[aid][index]` and
   `charm.groups[gid]` on every message (charm4py/charm.py:329-399) go away. (3.3)
4. **Sections are CkMulticast sections**, referred to by a 16-byte id that names the
   root PE's table entry; sends from other PEs and section callbacks are forwarded
   through the per-PE manager group, and `charm.combine` builds a deduplicated union
   section over arrays of one index type (what CkMulticast's multi-array sections
   support); combines that mix arrays with groups, or arrays of different index
   types, are not supported in the thin shim. `sections.py`, its spanning tree, its reduction tree and `SectionManagerExt`
   are deleted; constrained groups (`Group(C, onPEs=...)`) become a group plus a group
   section. (3.9)
   Decision 2026-11-08 (review F17): `charm.combine` is a deduplicated union
   materialized as its own CkMulticast section, not a composite reduced at the root;
   mixed array-and-group combines are dropped (3.9).
5. **Reductions use Charm++'s reducer table directly**, with the element type of each
   built-in reducer exported by CkDyn so the binding derives its conversion table
   instead of hand-maintaining three copies (ckreduction.h:277-282 warns about exactly
   this), and a `CkDynReducerInfo` entry costs no code in the binding. Binding-defined reducers go through one foreign reducer whose contributions
   carry a binding-assigned reducer id. (3.8)
6. **Future ids are carried in the envelope reference number, and libcharm is built with
   32-bit reference numbers** (`REFNUM_TYPE="unsigned int"`, a CMake cache variable,
   CMakeLists.txt:184). Today's mechanism is the same but the default `unsigned short`
   silently truncates a future id carried through a callback above 65535, although
   future ids are already 32-bit (threads.py:10). No change to `CkCallback` is
   needed. (3.7)
7. **Coroutines keep greenlets but take them from a per-PE pool** instead of creating
   one per invocation (entry_method.py:95-102). (4.5)
8. **Reconverse first; classic not at all for the thin design.** `classic` is frozen and
   takes fixes only (charm/AGENTS.md); porting CkDyn there is not cheap and the existing
   charm4py already works there. (6.1, 8.2)
9. **CkDyn is a Charm++ subsystem with its own C++ tests** (`tests/charm++/ckdyn`), so
   that `recharm` CI covers it without Python, and an internals chapter
   (`doc/internals/ckdyn.md`). That is the ownership answer. (3.16, 8.1)
10. **Projections and shrink/expand are requirements, not extensions** (Kale,
    2026-11-07). Every Python class is its own Charm++ chare index and every method its
    own entry index, so the `.sts` file names them (3.13); the libcharm in wheels is
    built with `--enable-tracing`. Shrink/expand needs only pup and `AtSync` from CkDyn,
    which the design has, plus the Converse half that reconverse lacks today (3.10, 8.4,
    step 8).
    Decision 2026-11-08 (review F13): parity needs two runtime changes, element
    indices for classes registered with `ndims` 0 and per-element trace events on the
    many-element broadcast path, and the Projections gate checks an array element
    index in the log (3.13).

Cost targets, measured as described in 7.4 on this Mac with the production reconverse
build and one process ("today" is the step 0 measurement of 2026-11-07, BENCHMARKS.md;
"floor" is the C++ 1D-array ping-pong on the same build, which no binding can beat):

| path (per message, one PE) | today | floor | target |
|---|---:|---:|---:|
| plain entry method, one int argument | 2.27 us | 0.18 us | 1.0 us |
| coroutine entry method, one int argument | 6.14 us | 0.18 us | 1.4 us |
| plain entry method, 1 MB numpy argument | 59.6 us | 0.18 us | 23 us (one 1 MB memcpy, 20.4 us, on the send side; none on receive) |

---

## 1. Scope and conventions

In scope: the full feature list of today's charm4py (the public API in
charm4py/docs and the acceptance suite `charm4py/tests` plus the examples listed in
`test_config.json`), plus the Charm++ features the analysis found missing: nodegroups,
checkpoint/restart, shrink/expand, entry-method attributes, message priorities,
zero-copy host and device buffers, per-entry Projections tracing and GPU load
attribution. Out of scope:
changing what the user writes, except where section 5.4 lists a change and its reason.

Conventions: `CkDyn*` names are the proposed C++ and C API; `_ckdyn.*` names are the
proposed nanobind module; "binding" means any Layer 3a, Python or otherwise; "object"
means the binding's object (a Python chare instance); "element" means the CkDyn C++
object that Charm++ manages. "Today" means charm4py `c5b288f` on Charm++ `2a2d1ac67`.

---

## 2. Architecture

### 2.1 The three layers and what each may know

| layer | knows about | must not know about |
|---|---|---|
| 1. Charm++ core | envelopes, scheduling, location, LB, reductions, multicast, checkpoint | CkDyn (except the four additions in 3.15) |
| 2. CkDyn (C++, in `src/ck-core`) | Charm++ core; a binding only through the hook table of 3.3 and the slot types of 3.4 | Python, pickle, numpy, greenlets, the GIL |
| 3a. `charm4py._ckdyn` (nanobind) | CkDyn's C API; CPython's object model; numpy/cupy/torch buffer protocols; greenlet | Charm++ headers other than `ckdyn.h`; the Python API classes (it sees callables and opaque objects) |
| 3b. charm4py Python package | `_ckdyn`; the user-facing API | message bytes, Charm++ ids beyond opaque handles |

The test of a good boundary: a Julia or Rust binding replaces 3a and 3b and touches
nothing in layer 2. Every place where today's Ext API exposes a Python-specific detail is listed in 3.15
with what replaces it.

### 2.2 The path of one message, today and after

Today, `proxy.recv(3)` from an array element on the same PE
(charm4py/chare.py:721-773, charmlib_cython.pyx:832-884, 496-508, ck.C:2683-2703,
ckarray.h:517-533, charmlib_cython.pyx:978-986, charm.py:359-378, 793-830,
entry_method.py:39-74):

1. Python builds `header = {'is_ray': False}` and the args list; `packMsg` walks the
   args for buffer-protocol objects, pickles `(header, [3])`.
2. `CkArraySend` copies the pickle into a `CkMarshallMsg` behind three ints.
3. Charm++ delivers to `ArrayElemExt::__entryMethod`, which parses the three ints and
   calls `ArrayMsgRecvExtCallback` with the array id, the index as an int array, the ep.
4. Cython builds a Python tuple for the index, wraps the bytes in a `ReceiveMsgBuffer`,
   calls `charm.recvArrayMsg`, which looks up `self.arrays[aid][index]`, unpickles,
   checks `'is_ray'` and `b'dcopy'`, looks up `self.entryMethods[ep]`, checks
   `when_cond`, and calls `em.run(obj, header, args)`, which calls
   `getattr(obj, name)(*args)` and then inspects `header` for `b'block'`.

After:

1. The proxy method (a C++ vectorcall function bound by the binding, 4.2) switches on
   the argument types: one `int` becomes one I64 slot. No dict, no pickle. It calls
   `CkDynSendArray(aid, 1, &idx, ep, msg, nullptr)`.
2. Charm++ delivers to `ArrayElemDyn::callEntry`, which calls
   `hooks.deliver(handle, ep, view)` with the stored handle.
3. The binding indexes its method table by `ep`, converts the one slot to a Python
   `int`, and calls the unbound method with vectorcall on `(handle_obj, 3)`. Flags in
   the message header say whether a result must be sent to a future.

The per-message work that remains is unavoidable in any design: a Charm++ local send
(0.18 us on this build, BENCHMARKS.md), one Python call, and argument conversion.

### 2.3 Threads, PEs and the interpreter

Charm++ on reconverse runs `+pe N` PE threads in one process with no communication
thread. Today charm4py launches one process per PE (`charmrun/start.py`
`reconverse_args`, surveys/survey-tests.md section 1) and never runs two PEs in one
interpreter, and the Cython module contains no GIL handling at all (grep of
charmlib_cython.pyx for `PyGILState`, `nogil`: no hits). The design keeps "one PE per
process" as the supported configuration and makes the binding correct for several PEs
per process:

- `_ckdyn.start()` releases the GIL (`Py_BEGIN_ALLOW_THREADS`) before calling
  `CkDynStart`, which never returns until exit.
- Every hook entry (`deliver`, `construct`, `pack`, ...) takes the GIL with
  `PyGILState_Ensure` and releases it on return. Greenlets are per OS thread, so each
  PE thread owns its own main greenlet and pool (4.5).
- With the GIL, several PEs in one process serialize on Python and only the C++ parts
  overlap; with a free-threaded interpreter (CPython 3.13t+, which nanobind supports)
  they run in parallel. That is how charm4py would use several PEs per process, and
  it needs no change in Layers 1 and 2. It is deferred to step 7 of the plan (7.1).

Decision 2026-11-08 (review F14): taking the GIL in every hook does not make several
PEs per process correct, because a PE that calls into the runtime from Python holds
the GIL while it may wait for a runtime lock that another PE holds inside a hook
(4.4). Until step 7 the rule is one PE per process, and `_ckdyn.start` refuses
`CmiMyNodeSize() > 1`; the step 7 options are in 8.3.

---

## 3. Layer 2: CkDyn

### 3.1 Placement, build and ABI

Files, all in the Charm++ repository on branch `ckdyn`:

| file | content |
|---|---|
| `src/ck-core/ckdyn.h` | the public API: C structs and `extern "C"` functions (this is what a binding includes), plus the C++ classes for C++ users |
| `src/ck-core/ckdyn.C` | implementation |
| `src/ck-core/CkDyn.ci` | `message CkDynMsg { char buf[]; }` and the `CkDynMgr` group (3.7) |
| `tests/charm++/ckdyn/` | a C++ "binding" and its tests (3.16) |
| `doc/internals/ckdyn.md` | the internals chapter required by charm/AGENTS.md |

CkDyn is compiled into `libck` unconditionally, on every target, not only when
`TARGET=charm4py`. The reason is ownership (8.1): a subsystem that only builds in a
Python-specific configuration is not tested by the Charm++ CI and breaks without
anyone noticing, which is what
happened to the Ext API (`CkWallTimerHook` declared and never defined,
`CkRegisterArrayMapExt` ignoring `emNamesStart`, surveys/survey-ext-api.md Table A and
B). The cost is about 2,500 lines of code in every libck; nothing in it runs unless
`CkDynRegisterClass` is called.

The binding-facing API is C: POD structs, function pointers, `extern "C"` functions,
and an API version constant

```c
#define CKDYN_API_VERSION 1   /* bumped on any incompatible change to ckdyn.h */
int CkDynApiVersion(void);    /* the binding checks it at import */
```

C linkage is what makes the layer language-neutral (a Julia `ccall` or a Rust `extern
"C"` block can use it) and what lets nanobind call it without pulling Charm++ headers
into the extension. `ckdyn.h` includes no Charm++ header: every type it uses is its own POD
(`CkDynTarget`, `CkDynCallback`, `CkDynSectionID`, the descriptors), and the few
runtime calls a binding needs besides CkDyn's own (PE numbers, exit, abort, printf,
topology, CCS, timed callbacks) have `CkDyn*` wrappers (3.17), as `CkMyPeHook` does
today (ck.C:2408). The C++ classes behind the API (`GroupDyn`, `ArrayElemDyn`, ...)
are declared in `ckdyn-impl.h` for C++ users and the tests and are not part of the
ABI.

### 3.2 Classes, entry tables and registration

A binding describes each class once, before start-up, through a table:

```c
typedef enum {
  CKDYN_MAINCHARE = 1, CKDYN_CHARE, CKDYN_GROUP, CKDYN_NODEGROUP, CKDYN_ARRAY, CKDYN_ARRAYMAP
} CkDynKind;

/* entry-method attributes; the names match the .ci attributes */
enum {
  CKDYN_EP_THREADED   = 1u << 0,  /* binding runs it on a coroutine; CkDyn passes the flag through */
  CKDYN_EP_INLINE     = 1u << 1,  /* CK_EP_INLINE: a local send runs the method at once */
  CKDYN_EP_EXPEDITED  = 1u << 2,  /* sends get CK_MSG_EXPEDITED */
  CKDYN_EP_IMMEDIATE  = 1u << 3,  /* CK_EP_IMMEDIATE (nodegroups only) */
  CKDYN_EP_NOKEEP     = 1u << 4,  /* always set by CkDyn; see 3.4 on message retention */
  CKDYN_EP_NOTRACE    = 1u << 5,  /* CK_EP_TRACEDISABLE */
  CKDYN_EP_APPWORK    = 1u << 6,  /* CK_EP_APPWORK */
  CKDYN_EP_EXCLUSIVE  = 1u << 7,  /* nodegroups: one at a time per branch */
  CKDYN_EP_REDNTARGET = 1u << 8,  /* may be the target of a reduction (documentation only) */
};

typedef struct CkDynEntryDesc {
  const char *name;     /* "Class.method"; stored by pointer for the program's lifetime */
  uint32_t    flags;    /* CKDYN_EP_* */
  uint32_t    user_id;  /* opaque to CkDyn, passed back on every delivery (e.g. index into the binding's method table) */
} CkDynEntryDesc;

typedef struct CkDynClassDesc {
  const char           *name;        /* "Class" */
  CkDynKind             kind;
  int                   ndims;       /* arrays only; 1..CK_ARRAYINDEX_MAXLEN */
  int                   num_entries; /* entries[0] is the constructor */
  const CkDynEntryDesc *entries;
  void                 *binding_data;/* opaque; passed to every hook for this class */
  const CkDynHooks     *hooks;       /* NULL: the table given to CkDynInit; a second binding in the same process gives its own (8.7) */
} CkDynClassDesc;

/* Returns the chare index. *first_ep receives the Charm++ entry index of entries[0];
   entries[i] has index first_ep + i. */
int CkDynRegisterClass(const CkDynClassDesc *desc, int *first_ep);
```

What `CkDynRegisterClass` does, per kind (the calls are the ones register.C:105-174 and
194-209 make today, generalized):

| kind | `CkRegisterChare(name, sizeof(X), type)` | base | constructor ep | extra eps registered by CkDyn |
|---|---|---|---|---|
| MAINCHARE | `MainchareDyn`, `TypeMainChare` | `CkIndex_Chare::__idx` | `CMessage_CkArgMsg::__idx`, flags 0, `CkRegisterMainChare` | migration ctor for checkpoint (3.10) |
| CHARE | `ChareDyn`, `TypeChare` | `CkIndex_Chare::__idx` | `CkDynMsg::__idx` | none |
| GROUP | `GroupDyn`, `TypeGroup` | `CkIndex_IrrGroup::__idx` (so `CkArrayMap` and `Group` both fit) | `CkDynMsg::__idx`, `CkRegisterDefaultCtor` | migration ctor (checkpoint) |
| NODEGROUP | `NodeGroupDyn`, `TypeNodeGroup` | `CkIndex_NodeGroup::__idx` | same | migration ctor |
| ARRAY | `ArrayElemDyn`, `TypeArray`, `CkRegisterArrayDimensions(idx, ndims)` | `CkIndex_ArrayElement::__idx` | `CkDynMsg::__idx`, `CkRegisterDefaultCtor` | migration ctor (`CkRegisterMigCtor`), `AtSync`, `ResumeFromSync` |
| ARRAYMAP | `ArrayMapDyn`, `TypeGroup` | `CkIndex_IrrGroup::__idx` | same as GROUP | none |

Every user entry `entries[i]`, `i >= 1`, is registered with
`CkRegisterEp(desc->entries[i].name, X::callEntry, CkDynMsg::__idx, chareIdx, ckflags)`
where `ckflags` is derived from `flags`: `CK_EP_NOKEEP` always (3.4), `CK_EP_INLINE`
for INLINE, `CK_EP_IMMEDIATE` for IMMEDIATE, `CK_EP_TRACEDISABLE` for NOTRACE,
`CK_EP_APPWORK` for APPWORK (register.C:55-60 shows how each flag lands in `EntryInfo`).
The name is what Projections shows (3.13); the binding passes `"Class.method"` rather
than today's bare method name (charm.py:497-498).

EXPEDITED and THREADED are not Charm++ entry flags. EXPEDITED is remembered by CkDyn
and turned into `CK_MSG_EXPEDITED` on every send to that ep (3.5). THREADED is passed
through in the delivery (the `ep_flags` argument of `hooks.deliver`) so the binding
runs the method on a coroutine; Charm++'s own `[threaded]` (a `CthThread` per invocation, generated by
charmxi) is unusable because a Cth thread cannot run interpreter frames
(ANALYSIS-2026-11-06.md, "Two things cannot be thin"). EXCLUSIVE on a nodegroup entry
maps to the lock CkDyn takes around the delivery (3.6).

**When registration happens.** Charm++ requires all registration to be complete before
`_registerDone()` (init.C:1658); CkDyn keeps the existing hook that replaces
`CkRegisterMainModule` (init.C:1621-1630) but renames and generalizes it:

```c
typedef struct CkDynHooks { ... /* 3.3 */ } CkDynHooks;
void CkDynInit(const CkDynHooks *hooks);          /* before CkDynStart; copies the table */
void CkDynStart(int argc, char **argv);           /* = StartCharmExt (ck.C:2312-2338); never returns */
/* hooks.register_classes() is called from _initCharm in place of CkRegisterMainModule();
   the binding calls CkDynRegisterClass for each class inside it. */
```

`_initCharm` calls `CkRegisterMainModule()` when the program has one and the hook
otherwise; the dispatch is a weak symbol check, not a `CMK_CHARM4PY` ifdef, so one
libck serves both executables and the shared library (6.1).

**Late registration** (interactive mode, `CharmRemote.registerNewChareType`,
charm.py:1215-1226) stays possible: register.C:48-53 compiles the "late entry method
registration" abort out under `CMK_CHARM4PY`; CkDyn replaces that with a run-time
switch `CkDynAllowLateRegistration()` that the binding calls once. The constraint that
`_entryTable` is one process-wide vector (register.h, `std::vector<EntryInfo*>`) means
late registration must not race with dispatch on another PE of the same process.
`CkDynRegisterClassLate(desc, first_ep)` therefore registers only when the process has
one PE (`CmiMyNodeSize() == 1`, the only configuration charm4py runs today, 2.3) and
aborts with a message otherwise; the binding broadcasts the request to every process
in the same order, as charm4py already does with a group broadcast
(charm.py:1215-1226). Supporting it with several PEs per process would need a
process-wide stop of dispatch and is left out until step 7 needs it. The only test is
`tests/interactive/test1.in`.

### 3.3 The hook table: how the runtime reaches the binding

Today thirteen callback setters exist, one per situation (ck.C:2340-2406,
surveys/survey-ext-api.md A2). CkDyn has one table, passed once:

```c
typedef struct CkDynTarget {          /* identifies one element */
  CkDynKind kind;
  int       chare_idx;
  union {
    struct { int pe; void *ptr; } chare;            /* MAINCHARE, CHARE: the two fields of CkChareID (charm.h:278-281) */
    struct { int gid; int pe_or_node; } group;      /* GROUP, NODEGROUP, ARRAYMAP */
    struct { int aid; int ndims; int index[CKDYN_MAX_DIMS]; } array;   /* CKDYN_MAX_DIMS = 6 */
  } u;
} CkDynTarget;
```

Indices are always `int` in the API; CkDyn converts to and from `CkArrayIndex`, which
stores 1D-3D indices as three ints and 4D-6D as six shorts (`CK_ARRAYINDEX_MAXLEN` is
3 ints, ckarrayindex.h:13), a detail the binding has to know today
(charmlib_cython.pyx:293-307 reads shorts when `ndims > 3`).

```c

typedef struct CkDynMsgView CkDynMsgView;   /* read-only view of a delivered message, 3.4 */

typedef struct CkDynHooks {
  int   version;                                                    /* CKDYN_API_VERSION */
  void *global;                                                     /* passed to the hooks that have no class: register_classes, readonly, reduce, future, error */
  void (*register_classes)(void *global);                           /* 3.2 */
  /* object lifecycle */
  void *(*construct)(void *binding_data, const CkDynTarget *t, const CkDynMsgView *ctor_msg);
  void  (*destruct)(void *binding_data, void *handle);              /* element deleted (ckDestroy, exit) */
  /* delivery */
  void  (*deliver)(void *binding_data, void *handle, int ep, uint32_t user_id,
                   uint32_t ep_flags, const CkDynMsgView *msg);
  void  (*deliver_many)(void *binding_data, void **handles, int n, int ep, uint32_t user_id,
                        uint32_t ep_flags, const CkDynMsgView *msg);  /* array broadcast on one PE, 3.6 */
  void  (*mainchare)(void *binding_data, void *handle_out[1], const CkDynTarget *t,
                     int argc, char **argv);                        /* mainchare constructor */
  /* pup, 3.10 */
  size_t (*pack_size)(void *binding_data, void *handle, int reason);
  void   (*pack)(void *binding_data, void *handle, int reason, void *buf, size_t len);
  void  *(*unpack)(void *binding_data, const CkDynTarget *t, int reason, const void *buf, size_t len);
  size_t (*pack_readonly_size)(void *global);
  void   (*pack_readonly)(void *global, void *buf, size_t len);
  void   (*unpack_readonly)(void *global, const void *buf, size_t len);
  /* reductions, 3.8: runs inside the reduction manager, so it has no class context */
  void  (*reduce)(void *global, uint32_t reducer_id, const CkDynMsgView **contribs, int n,
                  CkDynMsgBuilder *result);
  /* futures, 3.7 */
  void  (*future)(void *global, int pe, uint32_t id, const CkDynMsgView *value);
  /* sections, 3.9 */
  void  (*section_joined)(void *binding_data, void *handle, const CkDynSectionID *sid);
  /* array map, 3.6 */
  int   (*map_proc_num)(void *binding_data, void *handle, int aid, int ndims, const int *index);
  /* LB, 3.12 */
  void  (*resume_from_sync)(void *binding_data, void *handle);
  /* errors raised inside CkDyn on behalf of the binding (e.g. a send to a missing branch) */
  void  (*error)(void *global, const char *what);
} CkDynHooks;
```

`binding_data` is the per-class pointer from the class descriptor; `handle` is whatever
`construct` returned, stored in the CkDyn element and passed back verbatim. CkDyn never
dereferences a handle. `reason` for pup is one of `CKDYN_PUP_MIGRATE`,
`CKDYN_PUP_CHECKPOINT`, `CKDYN_PUP_SIZING` (the binding may cache the packed bytes
between the sizing and packing passes, as the current `tempData` global does,
charmlib_cython.pyx:1009-1023, but per element rather than per PE).

All hooks are called on the PE that owns the element, inside the Charm++ scheduler,
with the message still owned by the runtime (3.4 says how long).

### 3.4 The messages: `CkDynMsg` and `CkDynMcastMsg`

Two Charm++ message types with one payload format serve every dynamic entry method,
constructor message, reduction result and section multicast:

```c++
// CkDyn.ci:  message CkDynMsg { char buf[]; };   message CkDynMcastMsg { char buf[]; };
class CkDynMsg : public CMessage_CkDynMsg {        // layout-identical to CkMarshallMsg (ckmarshall.h:8-11)
public:
  char *buf;
};
class CkDynMcastMsg : public CkMcastBaseMsg, public CMessage_CkDynMcastMsg {
public:
  char *buf;
};
```

Why two types. `CkMulticastMgr::sendToSection` writes the section cookie and the ep
into the first bytes of the user message (ckmulticast.C:915-958, `msg->ep = ep;
msg->_cookie = s;`), so a message sent through CkMulticast must begin with
`CkMcastBaseMsg` (charm/doc/charm++/manual.rst:7015-7032, which is also why marshalled
entries cannot be multicast). The zero-copy receive code casts the user message to
`CkMarshallMsg` and parses the `CkNcpyBuffer` descriptors from the start of `msgBuf`
(ckrdma.C:721, 733, 745, 846, 952, 1119, 1164; ckrdmadevice.C:718, 733), so a message
that carries zero-copy slots must have the `char *` at offset 0 that `CkMarshallMsg`
has. No one layout satisfies both, and no message needs both: a section multicast of
a zero-copy slot is delivered as a copy (3.6). `CkDynMsg` carries every point-to-point
send, broadcast, constructor message and reduction result; `CkDynMcastMsg` only what
`CkDynSendSection` sends. `callEntry` tells them apart by `env->getMsgIdx()`, which is
also how it knows that a delivery came through a section (3.9); nothing depends on
`CkMcastBaseMsg::checkMagic`, which returns true unconditionally in production builds
(cksection.h:78-84). Both are allocated with the operator new that charmxi generates
(`new (&size, prio_bits) CkDynMsg`), which sets `buf` to the varsize area and runs the
`CkMcastBaseMsg` default constructor (`_cookie.val = NULL`, cksection.h:38); a raw
`CkAllocMsg` would do neither. Two of the 256 message indices (envelope.h:169) are
used. Both message classes carry a `CkMcastBaseMsg`-sized or pointer-sized prefix, a
few dozen bytes, smaller than the envelope.

Payload layout (`buf`), fields little-endian and fixed-width, 8-byte aligned regions:

```c
struct CkDynHdr {
  uint16_t magic_version;   // 0xDA01
  uint16_t ep;              // Charm++ entry index; the authoritative one (3.6 explains why the envelope's can be stale)
  uint16_t flags;           // CKDYN_MSG_* below
  uint16_t nslots;
  uint32_t user_id;         // the entry's user_id, or the reducer id for foreign-reducer contributions
  uint32_t tail_len;        // bytes of opaque tail
  uint32_t data_len;        // bytes of the inline data region
  uint32_t nbuf;            // number of NCPY/DEVICE descriptors
};
struct CkDynSlot {          // 24 bytes each, nslots of them, right after the header
  uint8_t  type;            // CKDYN_SLOT_*
  uint8_t  elem;            // element kind for BUFFER/NCPY/DEVICE: CKDYN_ELEM_{I8..I64,U8..U64,F16,F32,F64,C64,C128,BOOL,BYTE}
  uint8_t  ndim;            // 0..CKDYN_MAX_BUF_DIMS (32) for buffers; 0 for scalars
  uint8_t  bflags;          // bit 0: Fortran order; bit 1: read-only; bit 2: came from the binding's serializer out-of-band
  uint16_t tag;             // binding-defined subtype, opaque to CkDyn (Python uses it to tell str from bytes,
                            //   array.array from numpy from memoryview, so that every argument round-trips to its type)
  uint16_t reserved;
  uint32_t len;             // bytes (BYTES/BUFFER/NCPY/DEVICE/TAIL), else 0
  uint32_t reserved2;
  union {
    int64_t  i64; uint64_t u64; double f64;   // inline scalar value
    uint64_t off;                             // offset into the data region, or descriptor index
  } v;
};
// data region: inline bytes of BYTES slots; shape[ndim] (int64) for every buffer slot; copied BUFFER contents;
//              pupped CkDynCallback / CkDynSectionID values
// tail: tail_len opaque bytes
// descriptor region (CkDynMsg only): when nbuf > 0, `buf` starts with the prefix the generated marshalling code
//   writes for nocopy parameters (impl_num_rdma_fields, root node, the pupped CkNcpyBuffer/CkDeviceBuffer objects;
//   xi-Parameter.C:388-396), and CkDynHdr follows it, because CkRdmaIssueRgets parses from the start of msgBuf.
```

Slot types:

| type | meaning | who converts |
|---|---|---|
| `NONE` | the binding's null (`None`) | binding |
| `BOOL`, `I64`, `U64`, `F64` | inline scalar | binding; CkDyn uses them for reduction results |
| `BYTES` | small blob inline in the data region (strings, bytes) | binding |
| `BUFFER` | typed n-dimensional buffer copied into the data region; shape in the data region | binding; CkDyn builds one for a reduction result |
| `NCPY` | typed buffer carried by a `CkNcpyBuffer` descriptor; the data arrives by RDMA get into memory CkDyn allocates or into a buffer the receiver posted (3.14) | both |
| `DEVICE` | typed device buffer carried by a `CkDeviceBuffer` descriptor (3.14) | both |
| `FUTURE` | `(pe:int32, id:uint32)` of a binding future, inline | the binding on a send; CkDyn from the envelope's reference number on a callback delivery (3.7) |
| `CALLBACK` | a `CkDynCallback` (3.7) in the data region | both |
| `SECTION` | a `CkDynSectionID` (3.9) in the data region (section proxies as arguments) | both |
| `PROXY` | `(kind, id, ndims, index[])` of a dynamic object, in the data region | both |
| `TAIL` | marks that the opaque tail exists; `len` = tail_len | binding |

Header flags: `CKDYN_MSG_BLOCK` (a FUTURE slot follows that must receive the method's
return value), `CKDYN_MSG_BCAST`, `CKDYN_MSG_BCASTRET` (gather the return values of a
broadcast), `CKDYN_MSG_CREATION` (constructor message that must acknowledge creation),
`CKDYN_MSG_USESATSYNC` (constructor message of an array element created with
`useAtSync`), `CKDYN_MSG_RAY`, `CKDYN_MSG_REDRESULT` (built by CkDyn from a
`CkReductionMsg`). They replace the keys of today's header dict (`b'block'`,
`b'bcast'`, `b'bcastret'`, `b'creation'`, `'is_ray'`, `b'sid'`, `b'single'`,
`b'constrained'`, chare.py:337-345, 473-498, 544-569, 737-771, 796-827) and the
leading `char useAtSync` of today's constructor message (ck.C:2500).

Building a message (sender side):

```c
typedef struct CkDynMsgBuilder CkDynMsgBuilder;
typedef void (*CkDynBufDone)(void *done_arg);     /* the source buffer may be released */

typedef struct CkDynBufDesc {
  const void    *ptr;
  uint64_t       len;        /* bytes */
  uint8_t        elem;       /* CKDYN_ELEM_* */
  uint8_t        ndim;       /* 0..CKDYN_MAX_BUF_DIMS */
  uint8_t        bflags;     /* order, read-only */
  uint16_t       tag;        /* binding subtype */
  const int64_t *shape;      /* ndim entries */
  const int64_t *strides;    /* ndim entries, bytes; must describe a contiguous layout, CkDyn rejects otherwise */
} CkDynBufDesc;

/* group_dep: a CkGroupID index the message must wait for on delivery, or -1. It is
   given here and not at send time because the envelope reserves space for group
   dependencies at allocation (msgalloc.C:154, `GroupDepNum`). */
CkDynMsgBuilder *CkDynMsgBegin(int nslots_hint, size_t data_hint, size_t tail_hint, int prio_bits, int group_dep);
void  CkDynMsgSetFlags(CkDynMsgBuilder*, uint16_t flags);
void  CkDynMsgSetUserId(CkDynMsgBuilder*, uint32_t user_id);       /* the header's user_id (foreign reducer id) */
void  CkDynMsgAddNone(CkDynMsgBuilder*);
void  CkDynMsgAddBool(CkDynMsgBuilder*, int);
void  CkDynMsgAddI64(CkDynMsgBuilder*, int64_t);
void  CkDynMsgAddU64(CkDynMsgBuilder*, uint64_t);
void  CkDynMsgAddF64(CkDynMsgBuilder*, double);
void  CkDynMsgAddBytes(CkDynMsgBuilder*, const void*, uint32_t len, uint16_t tag);
void  CkDynMsgAddBuffer(CkDynMsgBuilder*, const CkDynBufDesc*);                         /* copies now */
void  CkDynMsgAddNcpy(CkDynMsgBuilder*, const CkDynBufDesc*, CkDynBufDone, void *arg);  /* zero-copy; 3.14 */
void  CkDynMsgAddDevice(CkDynMsgBuilder*, const CkDynDevDesc*, CkDynBufDone, void *arg); /* 3.14 */
void  CkDynMsgAddFuture(CkDynMsgBuilder*, int pe, uint32_t id);
void  CkDynMsgAddCallback(CkDynMsgBuilder*, const CkDynCallback*);
void  CkDynMsgAddSection(CkDynMsgBuilder*, const CkDynSectionID*);
void  CkDynMsgAddProxy(CkDynMsgBuilder*, const CkDynTarget*);
void *CkDynMsgReserveTail(CkDynMsgBuilder*, uint32_t len);   /* the binding writes its serialized bytes in place */
CkDynMsg *CkDynMsgFinish(CkDynMsgBuilder*, int ep);          /* one allocation with the generated operator new; ep goes in the header */
void  CkDynMsgAbandon(CkDynMsgBuilder*);
void *CkDynMsgPriorityPtr(CkDynMsg*);                        /* the prio_bits reserved at Begin, to fill in */
void  CkDynMsgSetQueueing(CkDynMsg*, int queueing);          /* CK_QUEUEING_* */
void  CkDynMsgFree(CkDynMsg*);                               /* a finished message that will not be sent */
```

Every send stamps the header's `ep` with the entry it targets (`CkDynSend*`,
`CkDynCallbackSend`), so the ep given to `CkDynMsgFinish` matters only for a
constructor message handed to `CkDynArrayInsert`, which takes the class from it.

The builder accumulates slot records and data in a per-PE growable scratch buffer
(owned by the `CkDynMgr` branch, 3.7) and copies once into the final message; a
message whose payload is only scalars (the ping-pong case) takes a path that allocates
the message first and writes into it directly. A builder must be finished or
abandoned before the next send on the same PE; the binding never holds two.

Reading a message (receiver side):

```c
typedef enum { CKDYN_MSGKIND_DYN, CKDYN_MSGKIND_MCAST, CKDYN_MSGKIND_REDRESULT, CKDYN_MSGKIND_SYS } CkDynMsgKind;
struct CkDynMsgView {
  CkDynMsgKind         kind;
  const CkDynHdr      *hdr;        /* NULL for REDRESULT and SYS views, which have synthesized slots */
  const CkDynSlot     *slots;
  int                  nslots;
  const char          *data;
  const char          *tail;
  uint32_t             tail_len;
  uint32_t             refnum;     /* the envelope's reference number, 0 if none */
  const CkDynSectionID *section;   /* non-NULL for an MCAST delivery, 3.9 */
  void                *msg;        /* the runtime message, for retain */
};
const void *CkDynViewBufferPtr(const CkDynMsgView*, int slot, const int64_t **shape_out);  /* BUFFER or completed NCPY */
const void *CkDynViewBytesPtr(const CkDynMsgView*, int slot);
void        CkDynViewCallback(const CkDynMsgView*, int slot, CkDynCallback *out);
void        CkDynViewSection(const CkDynMsgView*, int slot, CkDynSectionID *out);
void        CkDynViewProxy(const CkDynMsgView*, int slot, CkDynTarget *out);
void        CkDynMsgRetain(const CkDynMsgView*);   /* CmiReference(UsrToEnv(msg)); pointers into the message stay valid */
void        CkDynMsgRelease(void *msg);            /* CmiFree(UsrToEnv(msg)); see below */
```

**Ownership.** Every CkDyn entry is registered `CK_EP_NOKEEP`, so the runtime frees
the message after `callEntry` returns, in `CkDeliverMessageFree` (ck.C:719-729),
through the dealloc function of the message type registered for the entry, which is
CkDyn's. That function is the plain `CkFreeMsg` charmxi generates (xi-Message.C:248-263),
so it is also correct for the other message kinds that reach a CkDyn entry through
callbacks: a `CkReductionMsg` (no conditional fields) and a system message (which is
then `CmiFree`d instead of returned to the system-message pool; harmless, and the same
thing `ArrayElemExt::__AtSyncEntryMethod` avoids by hand at ckarray.h:540-543). A
binding that wants to keep a zero-copy view of a `BUFFER` or `NCPY` slot alive (a
numpy array over message memory) calls `CkDynMsgRetain` before returning; CmiAlloc
blocks are reference counted (charm_best_practices.md, "Message-memory rules"), so the
message then lives until the last `CkDynMsgRelease`. `CkDynMsgRelease` is
`CmiFree(UsrToEnv(msg))` and not `CkFreeMsg`: `CkFreeMsg` aborts in error-checking
builds when called on the in-flight nokeep message (msgalloc.C:74-79), which is exactly
what happens when a view is dropped inside the method that received it. This is what
makes the receive side of large arrays copy-free (today `rebuildNumpyArray` does
`a.copy()`, charm.py:1282-1285, and charmlib_cython.pyx:820 the same).

**Why not pickle as the format.** The slot format is what makes Layer 2
language-neutral and lets CkDyn itself produce messages (reduction results, section
cookies, future deposits) without calling back into the binding. Today
`CreateCallbackMsgExt` (ckcallback.C:234-305, charmlib_cython.pyx:1045-1129) does a
C++-to-Python-to-C++ round trip per callback, with a re-pickle for futures and
sections; with the slot format a reduction result is a header, one `BUFFER` slot
pointing at `CkReductionMsg::getData()`, and an optional `FUTURE` slot, built in C++.

### 3.5 Sends and send options

```c
typedef struct CkDynSendOpts {
  uint32_t flags;        /* CKDYN_SEND_INLINE | EXPEDITED | IMMEDIATE | NOTRACE (OR'd into CK_MSG_*); NULL opts = 0 */
} CkDynSendOpts;
/* Priority, queueing and group dependencies belong to the message, not the send:
   prio_bits and group_dep are given to CkDynMsgBegin (the envelope reserves space for
   both at allocation), the bits are written through CkDynMsgPriorityPtr and the
   strategy set with CkDynMsgSetQueueing. An integer priority is the Charm++ convention
   of prio_bits = 8*sizeof(int) with the int as the bitvector and CK_QUEUEING_IFIFO. */

void CkDynSendChare    (const CkDynTarget*, int ep, CkDynMsg*, const CkDynSendOpts*);
void CkDynSendGroup    (int gid, int pe,   int ep, CkDynMsg*, const CkDynSendOpts*);  /* pe == -1: broadcast */
void CkDynSendGroupMulti(int gid, int npes, const int *pes, int ep, CkDynMsg*, const CkDynSendOpts*);
void CkDynSendNodeGroup(int gid, int node, int ep, CkDynMsg*, const CkDynSendOpts*);  /* node == -1: broadcast */
void CkDynSendArray    (int aid, int ndims, const int *index, int ep, CkDynMsg*, const CkDynSendOpts*); /* index == NULL: broadcast */
void CkDynSendSection  (const CkDynSectionID*, int ep, CkDynMcastMsg*, const CkDynSendOpts*);         /* 3.9 */
void CkDynSendTarget   (const CkDynTarget*, int ep, CkDynMsg*, const CkDynSendOpts*);                  /* by kind */
```

Each maps onto the same core call the Ext functions use today (ck.C:2603-2728:
`CkSendMsg`, `CkSendMsgBranch`, `CkBroadcastMsgBranch`, `CkSendMsgBranchMulti`,
`CProxyElement_ArrayBase::ckSendWrapper`, `CkBroadcastMsgArray`) plus
`CkSendMsgNodeBranch`/`CkBroadcastMsgNodeBranch` for nodegroups, with the `opts`
argument that those functions already accept and the Ext API always passes as 0.
Priorities and queueing go through `CkSetQueueing` and the priority bits reserved at
allocation (`CkAllocMsg(idx, size, prio_bits)`, charm.h:112; `CkPriorityPtr`,
charm.h:121), the same as `CkAllocateMarshallMsgT` does for `CkEntryOptions`
(ckmarshall.h:24-40).

How each attribute is honored, in one place:

| attribute | at registration | at send | at delivery |
|---|---|---|---|
| threaded | flag kept in CkDyn's per-ep table | nothing | `ep_flags` tells the binding to use a coroutine |
| inline | `CK_EP_INLINE` (read only on the callback send path, ckcallback.C:100-218) | CkDyn adds `CK_MSG_INLINE`; accepted by `CkSendMsg` (ck.C:1800), `CkSendMsgBranch` (ck.C:2000), `CkSendMsgNodeBranch` (ck.C:2157); arrays go through `CkSendMsgArrayInline` (ckarray.C:1327); a local target runs at once and the delivery is not counted by quiescence (ck.C:1837) | binding must tolerate nested delivery (4.4) |
| expedited | flag kept | CkDyn adds `CK_MSG_EXPEDITED`, accepted by every send path above and by `CkSendMsgArray`/`CkBroadcastMsgArray` (ckarray.C:1315, 1610) | nothing |
| immediate | `CK_EP_IMMEDIATE`, nodegroups only (charmxi rejects it elsewhere, xi-Entry.C:60-61; `CkSendMsg` aborts on it under error checking, ck.C:1807-1811) | `CK_MSG_IMMEDIATE` | on classic the message may run on the communication thread; on reconverse immediate is a no-op (`CmiIsImmediate(msg)` is 0 and `CmiBecomeImmediate` is empty, contrib/reconverse/include/converse.h:1166-1168), so the attribute is accepted and has no effect there |
| nokeep | `CK_EP_NOKEEP` always (msgalloc.C:74-79: the entry must not free the message itself) | nothing | retain if a view must outlive the call |
| priority | nothing | `prio_bits`/`prio`/`queueing` in the send opts; the scheduler enqueues with `CqsEnqueueGeneral(queueing, priobits, prioPtr)` (ck.C:1514-1584) | nothing |
| exclusive | flag kept (nodegroups) | nothing | `NodeGroupDyn::callEntry` mirrors the code charmxi generates (xi-Entry.C:2197-2210): `CmiTryLock(__nodelock)`; if busy, copy the message and re-send it to this node with `CkSendMsgNodeBranch`; else deliver and `CmiUnlock` |
| notrace | `CK_EP_TRACEDISABLE` | nothing | nothing |

Two limits of the entry table matter for a binding that registers one ep per method:
`envelope::epIdx` is an `UShort` (envelope.h:199), so a process holds at most 65536
entries including the runtime's own, and `msgIdx` is a `UChar` (envelope.h:169), so at
most 256 message types exist, which is why CkDyn registers exactly one. Registration
happens on rank 0 of each process into that process's own tables (register.h:327-335
aborts otherwise under error checking), in the same order everywhere.

### 3.6 The chare kinds

All six classes share one shape: a stored `handle`, a `callEntry` static that the
entry table points at, a constructor that calls `hooks.construct`, a migration
constructor and a `pup` (3.10), and a per-section cookie map (3.9). Declared in
`ckdyn-impl.h`:

```c++
struct CkDynObj {                          // the part every kind shares (a member, not a base, to keep the Charm++ bases first)
  void *handle = nullptr;                  // the binding's object
  int   chareIdx;                          // index into the class table (binding_data lookup)
  std::map<CkDynSectionID, CkSectionInfo> sectionCookies;
  void pup(PUP::er &p);
};

class MainchareDyn : public Chare {
  CkDynObj d;
public:
  MainchareDyn(CkArgMsg *m);               // creates CkDynMgr (3.7; the CkMulticastMgr comes with step 2), then hooks.mainchare(...)
  MainchareDyn(CkMigrateMessage *m);       // +restart
  static void callEntry(void *msg, void *obj);
  void pup(PUP::er &p) override;
};
class ChareDyn : public Chare {            // a real singleton chare: CkCreateChare, not migratable
  CkDynObj d;
public:
  ChareDyn(CkDynMsg *m);
  static void callEntry(void *msg, void *obj);
};
class GroupDyn : public Group {
  CkDynObj d;
public:
  GroupDyn(CkDynMsg *m);                   // hooks.construct(bd, target, view(m)); handle stays NULL on a PE outside a constrained group
  GroupDyn(CkMigrateMessage *m);
  static void callEntry(void *msg, void *obj);
  void pup(PUP::er &p) override;
  ~GroupDyn();                             // hooks.destruct
};
class NodeGroupDyn : public NodeGroup {
  CkDynObj d;
public:
  NodeGroupDyn(CkDynMsg *m);
  NodeGroupDyn(CkMigrateMessage *m);
  static void callEntry(void *msg, void *obj);     // takes __nodelock for EXCLUSIVE eps (3.5)
  void pup(PUP::er &p) override;
};
class ArrayElemDyn : public ArrayElement {
  CkDynObj d;
public:
  ArrayElemDyn(CkDynMsg *m);               // usesAtSync from CKDYN_MSG_USESATSYNC
  ArrayElemDyn(CkMigrateMessage *m);
  static void callEntry(void *msg, void *obj);
  static void callAtSync(void *msg, void *obj);    // the reserved AtSync ep
  void ResumeFromSync() override;          // hooks.resume_from_sync
  void pup(PUP::er &p) override;           // 3.10
};
class ArrayMapDyn : public CkArrayMap {
  CkDynObj d;
public:
  ArrayMapDyn(CkDynMsg *m);
  static void callEntry(void *msg, void *obj);
  int procNum(int arrayHdl, const CkArrayIndex &idx) override;   // hooks.map_proc_num
};
class CkDynMgr : public CBase_CkDynMgr { ... };    // the per-PE manager group of 3.7 and 3.9 (generated from CkDyn.ci)
```

`callEntry(msg, obj)` finds the message kind from `env->getMsgIdx()` and the ep
from the CkDyn header for a `CkDynMsg`/`CkDynMcastMsg` and from `CkMessageToEpIdx(msg)`
(charm.h:512, ck.C:3306-3313, which returns the array-broadcast ep from
`getsetArrayBcastEp()` for broadcast envelopes) for a reduction or system message.
The header copy exists because the envelope's `epIdx` is stale on one path: in a
production build `CkSendMsgInline` to a local chare never calls `_prepareMsg`
(ck.C:1833-1840, under `CMK_CHARMDEBUG` only), the same defect that charm #4108 fixed
for `CkSendMsgBranchInline` (charm_best_practices.md, "Inline entry methods were
traced as dummy_thread_ep"). CkDyn never builds an `isendChare` callback (it passes
`forceInline = false` and no CkDyn chare ep is INLINE), so the reduction and system
paths are not affected; the one-line fix of preparing the envelope unconditionally in
`CkSendMsgInline` is listed as a core change in 3.15 anyway. Then a `CkDynMsg` is viewed
directly, a `CkReductionMsg` is converted (3.7), a system message (a callback with no
data, ckcallback.C `if (!msg) msg=CkAllocSysMsg()`) becomes an empty view. Then
`hooks.deliver(bd, handle, ep, user_id, ep_flags, &view)`.

| class | base | notes |
|---|---|---|
| `MainchareDyn` | `Chare` | ctor takes `CkArgMsg`, calls `hooks.mainchare` (argc/argv); registered as the one main chare. Has a migration ctor and pup so that checkpoint works (3.10) |
| `ChareDyn` | `Chare` | a real singleton chare, created with `CkCreateChare(cIdx, ctorEp, msg, &id, onPE)`, with `onPE = CK_PE_ANY` handed to the seed balancer; not migratable. `Chare(C)` and `Chare(C, onPE=p)` in charm4py create this (5.4 item 13), replacing today's one-element array (chare.py:22-33), which costs a group id, a CkArray branch and a location entry on every PE and a creation broadcast per singleton and therefore cannot serve programs that create billions of chares. The cost of one `ChareDyn` is the object plus a `CkChareID` (PE and pointer) and nothing on other PEs |
| `GroupDyn` | `Group` | one branch per PE; `CkCreateGroup`. For a constrained group (`onPEs`), the ctor message carries `CKDYN_MSG_CONSTRAINED` and the PE list as its last slot (`BYTES` of int32); branches not in the list keep `handle == NULL`, and a send to such a branch calls `hooks.error`. CkDyn also creates a group section over the listed PEs (3.9) and passes its `CkDynSectionID` to `construct` so the binding can build `thisProxy` |
| `NodeGroupDyn` | `NodeGroup` | one branch per process; `CkCreateNodeGroup`; `callEntry` takes `__nodelock` when the ep is EXCLUSIVE. A nodegroup entry may run on any PE of the process (charm_best_practices.md "A nodegroup branch entry method can run on ANY PE"). Decision 2026-11-08 (review F14): taking the GIL in every hook is not enough when a process has several PEs (lock-order inversion, 4.4), so until step 7 a process has one PE and every nodegroup entry runs on it |
| `ArrayElemDyn` | `ArrayElement` | ctor from `CkDynMsg` (`CKDYN_MSG_USESATSYNC` in the header flags, not a leading `char` as at ck.C:2500); migration ctor; `pup` (3.10); `ResumeFromSync` calls `hooks.resume_from_sync` (today: a self-send to a Python ep, charmlib_cython.pyx:1036-1043); `AtSync` is a plain C function `CkDynAtSync(target)` since the element is local |
| `ArrayMapDyn` | `CkArrayMap` | `procNum(arrayHdl, idx)` calls `hooks.map_proc_num(bd, handle, ...)`; `populateInitial` default |

Array broadcast delivery stays one call per PE: the charm4py-specific
`CkArrayBroadcaster::deliverAndUpdate` (ckarray.C:1506-1580, 1819-1847) becomes a
generic path selected by a `ChareInfo::isDynamic` bit that `CkDynRegisterClass` sets:
when the first local element is dynamic, the broadcaster collects the handles of the
elements whose bcast epoch matches and calls `hooks.deliver_many(handles, n, ...)` once.
The binding deserializes the tail once and calls n methods. The ZC-broadcast and
stable-locations branches that the charm4py path bypasses today stay bypassed for
dynamic arrays (a dynamic array's broadcast of an `NCPY` slot is delivered as a copy;
noted as a limitation in 8.4).

Creation API:

```c
typedef struct CkDynArrayOpts {
  int   ndims; const int *dims;          /* NULL dims: empty array, insert later */
  int   map_gid;                         /* -1: default map */
  int   uses_at_sync;                    /* elements start with usesAtSync */
  int   anytime_migration;               /* CkArrayOptions::setAnytimeMigration */
  int   static_insertion;
  int   bind_to_aid;                     /* -1 or an array to bindTo */
} CkDynArrayOpts;
/* The group the creation must wait for (the map; see 8.4 on the bindTo+map race) is
   the group_dep of the constructor message, given to CkDynMsgBegin. */

void CkDynCreateChare    (int chare_idx, int on_pe, CkDynMsg *ctor, CkDynTarget *out);   /* on_pe -1: seed balancer */
int  CkDynCreateGroup    (int chare_idx, CkDynMsg *ctor);   /* constrained: CKDYN_MSG_CONSTRAINED in the ctor flags and
                                                               the PE list as the last slot, BYTES of int32 */
int  CkDynCreateNodeGroup(int chare_idx, CkDynMsg *ctor);
int  CkDynCreateArray    (int chare_idx, CkDynMsg *ctor, const CkDynArrayOpts*);                    /* returns aid */
void CkDynArrayInsert    (int aid, int ndims, const int *index, int on_pe, CkDynMsg *ctor);   /* ctor finished with the class's first_ep */
void CkDynArrayDoneInserting(int aid);
void CkDynArrayDestroyElement(int aid, int ndims, const int *index);                               /* ckDestroy, local only */
```

The constructor message is a `CkDynMsg` like any other; `CkDynCreateArray` sets
`ArrayEltInitMsg` on it and goes through `CProxyElement_ArrayBase::ckCreateArray`
exactly as ck.C:2491-2516 does, with the options the Ext API has no way to pass
(`anytime_migration`, `static_insertion`, `bind_to`, `group_dep`). Creation
acknowledgement (`charm.awaitCreation`) is unchanged in mechanism: the constructor
message carries `CKDYN_MSG_CREATION` and a FUTURE slot, and each constructed element
contributes an empty reduction to that future; CkDyn does the contribution itself when
`construct` returns, so the binding's constructor path has nothing to do.

### 3.7 Callbacks into dynamic objects

A callback is described to CkDyn as a POD and converted to a `CkCallback` internally,
so the C API never exposes the C++ class:

```c
typedef enum { CKDYN_CB_TARGET = 1, CKDYN_CB_SECTION, CKDYN_CB_FUTURE, CKDYN_CB_IGNORE, CKDYN_CB_EXIT } CkDynCbKind;
typedef struct CkDynCallback {
  CkDynCbKind    kind;
  CkDynTarget    t;        /* TARGET: the element or collection; bcast selects broadcast */
  int            ep;       /* TARGET, SECTION */
  int            bcast;
  uint32_t       refnum;   /* non-zero: the binding future the result belongs to (TARGET, SECTION) */
  CkDynSectionID sid;      /* SECTION */
  int            pe;       /* FUTURE: the PE that owns the future */
  uint32_t       id;       /* FUTURE */
} CkDynCallback;           /* fixed size; carried in CALLBACK slots and pupped as bytes */
```

Conversion, done wherever CkDyn needs a `CkCallback` (contribute, QD, HAPI, creation
acknowledgement):

| kind | `CkCallback` built | notes |
|---|---|---|
| TARGET, element | `sendChare` / `sendGroup` / `sendNodeGroup` / `sendArray` with the existing constructors, `hasRefnum = (refnum != 0)` | the target entry is a CkDyn ep, so whatever message arrives is converted by `callEntry` (3.6) |
| TARGET, bcast | `bcastGroup` / `bcastNodeGroup` / `bcastArray` | same |
| SECTION | `sendGroup` to `CkDynMgr[creating PE].relay`, `refnum` = a slot in that branch's relay table holding `(sid, ep, refnum)` | the branch forwards the delivered message to the section's root as a section send (3.9); two hops, which is also today's path (chare.py:493-497) |
| FUTURE | `sendGroup` to `CkDynMgr[pe].deposit`, `refnum = id` | `deposit` calls `hooks.future(global, pe, id, view)` |
| IGNORE, EXIT | `CkCallback::ignore`, `CkCallback::ckExit` | |

Nothing marks a callback as "external": `callEntry` converts by message kind (3.6).
This deletes `isExtCallback`, the four Ext constructors, `CkCallbackSendExt` and
`CreateCallbackMsgExt` (ckcallback.h:125-157, 396-452; ckcallback.C:226-327, 522-551)
and removes the `CMK_CHARM4PY` replacement of `s_section` (ckcallback.h:125-143); the
C++ `bcastSection` callback is not used by CkDyn, because it holds pointers to the
section's element list (ckcallback.h:133-141), cannot be pupped
(`CkCallback::pup` has no case for it, ckcallback.C:461-520) and is refused by
`CkStartCheckpoint` (ckcheckpoint.C:815). The SECTION kind above is pup-able and
checkpoint-safe since it targets a group branch. For the same reason the binding's
checkpoint callback (3.10) must not be a TARGET on a main chare: `sendChare` callbacks
contain a pointer under `CMK_CHARE_USE_PTR` (ckcallback.C:555-560); the binding uses a
FUTURE callback, which targets a group.

Decision 2026-11-08 (review F11): a FUTURE callback alone cannot resume a restarted
program, because the future id belongs to the checkpointed run and no coroutine waits
on it after `+restart`. `charm.checkpoint(dir, resume)` therefore takes a `resume`
target that is an entry method of a group, an array element or the main chare. A
group or array target is an ordinary TARGET callback. A main-chare target is routed
through `CkDynMgr` by chare index, not by pointer, so the callback holds no
`CMK_CHARE_USE_PTR` pointer and survives the checkpoint. On the original
(non-restarted) run the call also deposits to an optional future so that a coroutine
`main` can continue after the checkpoint completes.

Conversion of a `CkReductionMsg` in `callEntry`: a view of kind `REDRESULT` with one
synthesized `BUFFER` slot whose `elem` comes from the reducer's element kind (3.8) and
whose pointer is `redMsg->getData()` (no copy; `view.msg` is the reduction message,
retained the same way), plus a `FUTURE` slot when the envelope carries a reference
number. For a foreign reducer the data is itself a CkDyn payload (3.8) and the view is
built over it as a `DYN` view, with the reference number in the view's `refnum` field
and no synthesized FUTURE slot (the slots are the reducer's). A system message (a callback sent with no data; `CkCallback::send`
allocates one, ckcallback.C:373) becomes a `SYS` view with no slots except the
optional `FUTURE`.

**The reference number.** A `CkCallback` with `hasRefnum` sets the envelope's `ref`
field on whatever message it sends (`CkSetRefNum`, ckcallback.C:~400-430;
`envelope::ref`, envelope.h:196, 226-227). That is the mechanism charm4py uses today
(the Ext constructors at ckcallback.h:398-452 set `refnum = fid`), and `callEntry`
keeps it: a non-zero reference number on a `REDRESULT` or `SYS` delivery means "this
result belongs to future `ref` on the PE the callback targeted", and `callEntry`
writes a `FUTURE` slot `(CkMyPe(), ref)`. The rule is safe only because CkDyn controls
every path that sets `ref` on a message to a CkDyn entry: its own `CkCallback`
constructions, and section reductions, where `CkMulticastMgr` sets `ref` from the
`userData` argument of `contribute` (ckmulticast.C:1436) whose default is -1
(ckmulticast.h:98-110); `CkDynContributeSection` always passes `userData = refnum`
(0 when there is no future), never the default. `CkFuture`'s own use of `ref`
(ckfutures.C:440-570) never targets a CkDyn entry.

What changes is the width: `CMK_REFNUM_TYPE` is `unsigned short` by default, so a
future id carried in a callback is silently truncated above 65535 although future
ids themselves are already 32-bit (`FIDMAXVAL = 4294967295`, threads.py:10, 319-322;
chare.py:126-130 works around the width for `allreduce` ids). libcharm is built with
`REFNUM_TYPE="unsigned int"` (`./build ... --with-refnum-type="unsigned int"`,
buildcmake:427-428, 762; CMakeLists.txt:184, 368), which widens `envelope::ref` by two
bytes in that build only and changes no code. The binding checks
`CkDynRefnumBytes() == 4` at import.

**Collective futures** (`allreduce`, chare.py:124-147) need an id every participant
computes identically before any message is exchanged. For a whole collection it is
the contributor's reduction number (`Group::getRedNo()`, `ArrayElement::getRedNo()`),
available as `CkDynGetReductionNumber(target)` (today ck.C:2557-2571); for a section
it is the section cookie's `redNo` (`CkSectionInfo::redNo`, cksection.h:36), available
as `CkDynSectionRedNo(target, sid)` from the per-section cookie each member stores
(3.9), which is what today's `_scookies[sid]` counter approximates (chare.py:131-142).
Decision 2026-11-08 (review F17): a combined section (`charm.combine`) is its own
CkMulticast section with its own cookie (3.9), so its members also take the shared id
from `CkDynSectionRedNo` and `allreduce` on a combined section needs no other
mechanism.

**`CkDynMgr`.** A group in `CkDyn.ci`, one branch per PE, with these roles: it owns the
per-PE scratch buffer of the message builder; its `deposit` entry delivers futures
(`CkDynFutureSet(pe, id, msg)` sends a `CkDynMsg` to `CkDynMgr[pe].deposit`, which calls
`hooks.future`; today this is the user-level group entry
`CharmRemote._future_deposit_result`, charm.py:1185-1186); its `relay` entry serves
section callbacks; its `sectionSend`, `sectionInit` and `compositeContribute` entries
implement sections (3.9); and on each process's rank 0 it owns the late-registration
path (3.2). It is created, together with the one `CkMulticastMgr` CkDyn uses, by the
`MainchareDyn` constructor on PE 0 before the binding's main-chare hook runs, so
"groups created in the main chare constructor exist on every PE before any other
message is delivered" (charm_best_practices.md, "Two creation regimes") covers both;
their group ids are part of the `ckdyn_ro` readonly. A C++ program that uses CkDyn
without `MainchareDyn` calls `CkDynCreateManagers()` from its own main chare
constructor.

### 3.8 Reductions

```c
typedef struct CkDynReducerInfo { const char *name; int type; uint8_t elem; } CkDynReducerInfo;
int  CkDynReducers(const CkDynReducerInfo **out);       /* all built-in reducers with their element kind */
int  CkDynForeignReducerType(void);                      /* the one CkReduction::reducerType for binding-defined reducers */

typedef struct CkDynRedData { const void *data; uint32_t len; int type; } CkDynRedData;  /* type: CkReduction::reducerType */
void CkDynContribute(const CkDynTarget *me, const CkDynRedData *d, const CkDynCallback *cb);
/* CkDynContributeSection: 3.9 */
/* a foreign contribution's payload comes from a finished message: */
const void *CkDynMsgPayload(CkDynMsg *m, uint32_t *len);
```

`CkDynContribute` finds the local element from the target (as
`getExtContributor<>` does, ckreduction.C:1893-1912) and calls its
`contribute(len, data, type, cb)`; nodegroups contribute on `NodeGroup`. The reducer
table is built by `initReducerTable` (ckreduction.C:1734-1858) and CkDyn exports it
with element kinds derived from a static table keyed by name (`sum_int` -> I32 and so
on; `CkReduction::set`, `concat`, `statistics`, `tuple` are exported with elem BYTE and
left to the binding). `reduction.py`'s `populateConversionTables` (reduction.py:201-300)
and `CkReductionTypesExt` (ckreduction.h:274-350, duplicated at
charmlib_cython.pyx:59-161) are deleted; the binding builds its numpy-dtype-to-reducer
map from `CkDynReducers` at import.

**Foreign reducer.** One reducer type `CkDyn::foreign` is appended to the table (as
`external_py` is today, ckreduction.C:1853-1856). A contribution to it is a `CkDynMsg`
payload (header + slots + tail) whose `user_id` is the binding's reducer id; the
Charm++ reducer function calls `hooks.reduce(global, reducer_id, views, n, builder)` and
turns the builder's result into `CkReductionMsg::buildNew`. The reducer id in the
header replaces today's `custom_reducer` name inside the pickle
(reduction.py:348-351, charmlib_cython.pyx:1132-1163), and the per-contribution
`loads` plus the `assert customReducer == currentReducer` become a header comparison.
Registration order is not a problem because the Charm++ reducer type is one fixed
entry; binding reducers may be added at any time, on every PE in the same order (the
binding's responsibility, as today with `charm.addReducer`).

`gather` and the broadcast-with-return path (`bcastret`, entry_method.py:63-74) stay
foreign reducers in the binding; the `nop` reducer with no data maps to a contribute
with `len 0`, as `CkExtContribute` does (ckreduction.C:1929-1934).

### 3.9 Sections over CkMulticast

```c
typedef struct CkDynSectionID {      /* 16 bytes, POD, copyable, carried in SECTION slots */
  int32_t  root_pe;                  /* PE whose CkDynMgr branch holds the CkSectionID and roots the multicast tree */
  int32_t  idx;                      /* index in that branch's section table */
  int32_t  coll_id;                  /* aid or gid of the collection; -1 for a combined section */
  uint8_t  kind;                     /* CKDYN_SECTION_ARRAY, _GROUP; 0 for a combined section */
  uint8_t  reserved;
  uint16_t gen;                      /* generation, so a stale id is detected after delete */
} CkDynSectionID;

/* Creation runs on the calling PE, which becomes root_pe. `done` fires when every member has stored its cookie. */
void CkDynSectionCreateArray(int aid, int nelems, int ndims, const int *indices /* nelems*ndims */, int bfactor,
                             const CkDynCallback *done, CkDynSectionID *out);
void CkDynSectionCreateGroup(int gid, int npes, const int *pes, int bfactor, const CkDynCallback *done, CkDynSectionID *out);
void CkDynSectionCombine(int nparts, const CkDynSectionID *parts, const CkDynCallback *done, CkDynSectionID *out);
void CkDynSectionDelete(const CkDynSectionID*);
int  CkDynSectionRedNo(const CkDynTarget *me, const CkDynSectionID*);                  /* 3.7 */
void CkDynSendSection(const CkDynSectionID*, int ep, CkDynMcastMsg*, const CkDynSendOpts*);
void CkDynContributeSection(const CkDynTarget *me, const CkDynSectionID*, const CkDynRedData*, const CkDynCallback *cb);
```

Why the id is a reference and not the `CkSectionID` itself: a `CkSectionID` holds
`std::vector<CkArrayIndex> _elems` and `std::vector<int> pelist` (cksection.h:112-121),
so it has no fixed size and would carry the whole member list in every message; and
its cookie's `val` is an `mCastEntry*` valid only on the PE that called `setSection`
(ckmulticast.C:923), while a send from any other PE takes the "root migrated" path
and prints "Warning: Multicast not optimized after multicast root migrated" every time
(ckmulticast.C:955-958). charm4py programs send to section proxies from arbitrary
PEs, and today's implementation already routes every such send to the root's
`SectionManager` (chare.py:493-497). So the `CkSectionID` lives in the root PE's
`CkDynMgr` table and everything else refers to it by `(root_pe, idx)`.

**Creation.** On the root, `CkDynSectionCreateArray` builds
`CProxySection_ArrayBase(aid, indices, nelems)`, delegates it to the `CkMulticastMgr`
(`ckSectionDelegate`), which runs `setSection`/`prepareCookie` for arrays
(ckmulticast.h:56, ckmulticast.C:401-424) or `prepareGrpCookie` for groups
(ckmulticast.C:426-440, through `initDelegateMgr`), stores the proxy in the table and
returns the id. CkMulticast requires each member to hold a `CkSectionInfo` cookie for
the section, obtained from a multicast message with `CkGetSectionInfo`
(manual.rst:7063-7067); `CkMulticastMgr::contribute` aborts on a member without one
(ckmulticast.C:1270-1271). So creation ends with one multicast of a CkDyn-reserved
entry, `__ckdyn_section_init`, which `CkDynRegisterClass` registers on every dynamic
class (3.2): each member stores the cookie in its per-section map keyed by the
`CkDynSectionID`, calls `hooks.section_joined(handle, sid)` so the binding can record
membership, and contributes an empty section reduction whose callback is `done`. That
reduction is the "section is ready" signal that `charm.split` waits for today
(`createSectionDown`'s completion, sections.py:35-55).

**Sends.** On the root PE, `CkDynSendSection` calls `CkMulticastMgr::ArraySectionSend`
or `GroupSectionSend` (ckmulticast.h:119, 128) with the `CkDynMcastMsg`; from any other
PE it sends the message to `CkDynMgr[root_pe].sectionSend(idx, ep)`, one extra hop.
On delivery, `callEntry` sees the `CkDynMcastMsg` index, refreshes the member's stored
cookie with `CkGetSectionInfo(cookie, msg)` (as the manual requires, so that the
cookie's `redNo` tracks the section's reductions), and sets `view.section`.
Decision 2026-11-08 (review F17): routing sends from non-root PEs through
`CkDynMgr[root_pe]` is accepted for step 2. It costs one extra hop per multicast, and
many senders to one section concentrate their traffic on the root PE.

**Reductions.** `CkDynContributeSection` looks up the member's stored cookie and calls
`CkMulticastMgr::contribute(len, data, type, cookie, cb, userData = refnum)`
(ckmulticast.h:105; 3.7 on `userData`). A member of several sections has one cookie
per section (tests/sections/multirand-split-combine.py:46-53 and callbacks.py:42-55
rely on that), which is what the map provides.

**Composites** (`charm.combine`). `tests/sections/multirand-split-combine.py:120-160`
combines sections drawn from both arrays and groups, and CkMulticast's cross-array
sections require arrays of one type (manual.rst:7126-7137). A composite is therefore a
CkDyn construct: the root's table entry lists the part ids; a send fans out one copy
per part (each to that part's root); a reduction has each part reduce to its own root,
whose `CkDynMgr.compositeContribute(composite, gen)` entry counts the `nparts`
partial results and reduces them once more, with the built-in reducer function from
`CkReduction::reducerTable()` or `hooks.reduce` for a foreign reducer, before calling
the callback. That is a reduction tree of depth two in CkDyn, against the full tree
and buffering logic in `sections.py`. Parts that are themselves composites are
flattened at combine time.

Decision 2026-11-08 (review F17), replacing the composite mechanism of the previous
paragraph: `combine` is a union, deduplicated when the combined section is created, so
an element that belongs to two parts is a member once and contributes once. The
combined section is materialized as its own CkMulticast section, using Charm++'s
existing support for multi-array section proxies rather than a list of parts; it
therefore has its own cookie on every member and its own reduction number, which is
the shared id `allreduce` needs (3.7). `CkDynSectionCombine` keeps its signature and
builds that section. A multi-array section proxy covers arrays of one index type, so
`combine` accepts only array sections whose arrays share an index type; a combine
that mixes group sections with array sections, or arrays of different index types,
raises an error in the thin shim (tests/sections/multirand-split-combine.py:120-160
does the former and is adapted to combine only its array sections). This follows
Kale's principle that charm4py supports what Charm++ supports; it is a behavior
change listed in 5.4. Kale's reservation, recorded here: combine-style constructs are departures from Charm++; in the long term either
Charm++ adopts them or charm4py supports only what Charm++ has.

Status 2026-11-08 10:00 PST, step 2 implemented on `ckdyn-dev` (e05d5cc0b, a2f12a613,
2fa8a5adf, 74ed0ff60; doc/internals/ckdyn.md "Sections"). Seven C++ test phases pass at
`+pe 1`, `+pe 4 +balancer RotateLB +randomized_msgq` (12 seeds), and `lcrun -n 2` with
and without RotateLB; a stale send aborts with the generation message. Deviations from
the text above, each kept: (1) the init entry is registered once on the base array and
group types, not per class, because CkMulticast sends one entry index to every
sub-section of a combined section; (2) a combined section has `coll_id -1`, `kind 0`,
since remote parts' arrays are unknown when the id is returned; (3) a combine over
arrays of different classes is accepted at creation and aborts on the first user send;
(4) relay slots for SECTION callbacks live on the PE that builds the callback; (5)
cookies and the section table are not checkpointed, so sections do not survive
`+restart` and the Python layer recreates them; (6) `CkDynSendSection` ignores send
options for now; (7) new `CkDynMsgFinishMcast` builds the multicast message. A
CkMulticast defect was found and fixed in e05d5cc0b (cross-array section results lost
their reducer type, user flag and callback in `XArraySectionReducer::finalReducer`);
that fix is independent of CkDyn and is a candidate for its own small recharm PR with a
plain C++ test. Relay slots are being re-keyed by (section, ep) with the refnum taken
from the arriving message, so the table is bounded (done in c81fbeecc: key (section, ep);
reduction results carry the refnum as the reduction user flag; CkDynMsg sends to a
SECTION callback go directly through `CkDynSendSection`; only QD and checkpoint callbacks
to a section keep one slot per refnum, because a Charm++ callback's send overwrites the
envelope refnum with the slot number). Step 4b (Python sections, `thin` e1de9d6, SPEC-4b.md)
exercised this API with no CkDyn defect found; one API gap: a built-in reduction result
relayed to a section entry is indistinguishable from a user send of a 0-d array (the
binding tags 0-d arrays on send; a CkDyn header flag set by the relay would remove the
need), and `CkDynSectionCreateGroup` needs a local branch of the group on the calling PE.

**Migration.** CkMulticast rebuilds its spanning tree when members migrate
(`needRebuild`, ckmulticast.C:941-951), so an element in a section may migrate; today
charm4py aborts in that case (charm.py:742-744). The per-section cookie map is part of
the element's pup (3.10).

What this removes: `sections.py` (201 lines: tree build, buffering until creation, a
reduction tree with `RedInfo`), `charm.split`/`charm.combine`'s location-gathering
reduction (`_getSectionLocations_`, `_sectionloc`, chare.py:179-225,
sections.py:192-201), the internal use of `getTopoSubtrees`
(charmlib_cython.pyx:652-689; the public `charm.getTopoSubtrees` stays for
tests/topo), `SectionManagerExt` and `CkForwardMulticastMsg`, the `s_section`
replacement in `CkCallback` and `CkStartQDExt_SectionCallback`. Section membership
functions (`section_func`) still run on the elements (5.2), but the vote returns
indices to the root, which calls `CkDynSectionCreate*`.

Reconverse caveat: CkMulticast's retire/retrieveCookie ordering dependence was fixed in
charm #4131 (charm_best_practices.md "Reconverse does not keep per-sender order"); the
C++ test in 3.16 runs under `+randomized_msgq` for that reason.

### 3.10 Pup: migration, checkpoint and restart, readonlies

One mechanism serves all three because Charm++ already drives them through the same
virtual `pup`:

```c++
void ArrayElemDyn::pup(PUP::er &p) {
  ArrayElement::pup(p);                       // usesAtSync, LB state (ckarray.h:560)
  p | sectionCookies;                         // the per-section CkSectionInfo map (3.9)
  int reason = p.isCheckpoint() ? CKDYN_PUP_CHECKPOINT : CKDYN_PUP_MIGRATE;   // pup.h:250-280
  if (p.isUnpacking()) {
    size_t n; p | n;
    if (auto *m = dynamic_cast<PUP::fromMem*>(&p)) {          // migration: hand the binding a pointer, no copy
      handle = hooks.unpack(bd, &target, reason, m->get_current_pointer(), n);   // as ckarray.h:576-579 does today
      m->advance(n);
    } else {                                                   // disk or network pupper: through a scratch buffer
      std::vector<char> tmp(n); p(tmp.data(), n);
      handle = hooks.unpack(bd, &target, reason, tmp.data(), n);
    }
  } else if (p.isSizing()) {
    size_t n = hooks.pack_size(bd, handle, reason | CKDYN_PUP_SIZING);   // the binding may cache the packed bytes
    p | n; p((char*)nullptr, n);                               // the sizer only counts
  } else {
    size_t n = hooks.pack_size(bd, handle, reason);
    p | n; char *scratch = CkDynScratch(n); hooks.pack(bd, handle, reason, scratch, n); p(scratch, n);
  }
}
```

`hooks.pack_size` may return `CKDYN_PACK_REFUSED` (`(size_t)-1`); the `pup` then calls
`hooks.error` with the object's target and reason and `CkAbort`s. The binding uses it
for an element that cannot be packed in that state, which for migration means one with
a suspended coroutine (4.5). Charm++ offers no alternative: a `[threaded]` C++ entry
method suspended across a migration is the same error unless isomalloc stacks move
it, and no such mechanism exists for interpreter frames.

`GroupDyn`, `NodeGroupDyn` and `MainchareDyn` have the same `pup` and a migration
constructor, which is what `CkStartCheckpoint(dirname, cb)` (ckcheckpoint.h:57) needs
to pack every group, nodegroup and the main chare: the restore path requires a
migration constructor for the main chare (ckcheckpoint.C:443-466) and aborts on a group
without one (ckcheckpoint.C:574-576). Today's `MainchareExt` and `GroupExt` register
none, and `ArrayElemExt::pup` and `ReadOnlyExt::_roPup` cast the pupper to
`PUP::fromMem` (ckarray.h:576, ck.C:2426), which holds for migration but not for a
disk pupper, so checkpoint and restart cannot work in the current charm4py at all
(surveys/survey-charm-apis.md section 10). The `pup` above branches on
the pupper's type: a `PUP::fromMem` (migration) hands the binding a pointer into the
buffer with no copy; any other pupper goes through a scratch buffer.
Readonlies: CkDyn registers one readonly `"ckdyn_ro"` whose pup function calls
`hooks.pack_readonly*` / `unpack_readonly`, replacing `ReadOnlyExt` (ck.C:2411-2432,
register.C:239-242); the same pup serves the start-up broadcast and the checkpoint
file. A `+restart dir` run then recreates every object through `unpack` with reason
CHECKPOINT; the binding's `unpack` for a mainchare or group must create the Python
object without running `__init__`, which charm4py already does for migration
(charm.py:385-390). The API addition on the Python side is `charm.checkpoint(dir,
callback)` (5.4). In-memory double checkpoint (`CkStartMemCheckpoint`) works through
the same pups and is listed as untested in 8.4.

Decision 2026-11-08 (review F11): the Python call is `charm.checkpoint(dir, resume)`,
with the `resume` target and the optional future of 3.7; after `+restart` the runtime
invokes `resume`, which is where a program's control flow continues, since the
suspended `main` coroutine is not restored (4.5 item 4). `tests/checkpoint/restart.py`
exercises `resume` after `+restart`. Singletons stay one-element arrays, as built in
step 4a (8.6 item 1), so a restart with a different PE count, which shrink/expand is,
loses no singleton: arrays are redistributed, while a `ChareDyn` restored per PE by
`CkPupChareData` would be dropped with its PE. Which PE hosts a real singleton chare
whose PE no longer exists is postponed (8.6 item 1).

Shrink/expand (manual.rst:11666-11730) is checkpoint plus load balancing plus a
relaunch: a CCS command received by `LBManager` (ck-ldb/manager.C:16-49) sets
`pending_realloc_state`; at the next `AtSync` step `CentralLB::CheckForRealloc`
(CentralLB.C:1111-1127) calls `CkStartCheckpoint(_shrinkexpand_basedir, cb)`; the
processes exit through `ConverseCleanup` (init.C:717-719) and are relaunched with the
new count, where `CkResumeRestartMain` restores readonlies, groups and the main chare
from the checkpoint and the arrays are redistributed by the balancer. What it asks of
the objects is exactly what checkpoint asks: a migratable main chare with a pup
(manual: `mainchare [migratable] Main`), pups on groups and array elements, and
`AtSync`/`ResumeFromSync`. `MainchareDyn`, `GroupDyn`, `NodeGroupDyn` and
`ArrayElemDyn` provide all of it, and `CkDynRegisterClass` registers every main chare
as migratable. CkDyn therefore adds nothing for shrink/expand except the build option
(`--enable-shrinkexpand`, CMakeLists.txt:180, 374) and one rule: with the option on,
`registerExitFn` aborts (init.C:1934-1941), so CkDyn registers no exit function and
the binding's at-exit work runs from the `CkExit` path instead. The binding side needs
the CCS server option (`++server`, already supported) and a balancer; the command
itself comes from the external CCS client as for C++ programs. The Converse half of the
feature (`Cmi_isOldProcess`, `Cmi_myoldpe`, `ConverseCleanup`, the relaunch with the
new process count) lived in the classic netlrts machine layer and charmrun, which
charm #4056 removed; reconverse has no implementation, so on `recharm` the option does
not link today (8.4). That is a reconverse work item, step 8 of the plan, independent
of CkDyn.

### 3.11 Quiescence

```c
void CkDynStartQD(const CkDynCallback *cb);   /* CkStartQD(CkCallback built from cb, 3.7); refnum carries the future */
```

The four `CkStartQDExt_*` functions (ck.C:2583-2601) collapse into this one. There is
no blocking `CkWaitQD` for bindings: a binding coroutine cannot block the PE, so
`charm.waitQD()` stays "start QD with a future, then wait on the future"
(charm.py:867-871). The reconverse-specific QD hang noted in charm_best_practices.md
("CkWaitQD/CkStartQD can hang on RECONVERSE even when the system is genuinely idle")
is a runtime issue outside this design; `tests/qd/qd.py` is the gate that will show it.

### 3.12 Load balancing hooks and GPU attribution

```c
void CkDynAtSync(const CkDynTarget*);                 /* ArrayElement::AtSync() on the local element */
void CkDynSetMigratable(const CkDynTarget*, int);      /* ck.C:2573-2581 */
void CkDynMigrateMe(const CkDynTarget*, int to_pe);    /* ck.C:2539-2549 */
void CkDynLBInstrument(int on);                        /* LBTurnInstrumentOn/Off, LBManager.h:610-611 */
void CkDynSetObjLoad(const CkDynTarget*, double secs); /* UserSetLBLoad-style override, for bindings that time themselves */
/* GPU attribution, see below */
void CkDynCallstackPush(const CkDynTarget*);
void CkDynCallstackPop(void);
```

`CkDynSetMigratable` and `CkDynMigrateMe` exist today in Ext form (ck.C:2539-2549,
2573-2581); `AtSync` is today a reserved entry (`__AtSyncEntryMethod`, ckarray.h:535-544)
reached by a self-send and becomes a direct call; `LBTurnInstrumentOn/Off` are general
LB API. Attribution of GPU work: Charm++ charges the
wall time of an entry method to the object that runs it through the running-object
stack, `CkCallstackPush(Chare*)` / `CkCallstackPop(Chare*)` (ck.C:631-674), which
stop timing the previous object, push the new one and start its timer; under
`CMK_LBDB_ON && CMK_CUDA` the same two functions push and pop a CUPTI external
correlation id (`hapiCuptiPushObjCorrelation`, ck.C:612-613, 640, 668), so every kernel
launched until the matching pop is charged to that chare's `LDObjData::gpuTime`
(lbdb.h:241-264). `CkDynCallstackPush/Pop` call them with the CkDyn element for the
target. Two uses by a binding: a coroutine that resumes after a future and launches a
kernel is no longer inside its entry method's bracket, so the binding pushes before
the launch and pops after; and `deliver_many` (3.6) pushes and pops around each
element's call so that a broadcast is attributed per element, which today's batched
Python broadcast does not do (ckarray.C:1506-1560 calls one Python function for all
elements with no running object set per element, so neither the load balancer nor
Projections sees per-element time for broadcasts). HAPI completion (3.14,
`CkDynHapiAddCallback`) delivers through a callback whose target is the element, so the
completion runs inside that element's bracket. No `hapiSetObjectLoad`-style override
exists in this tree; `CkDynSetObjLoad` is implemented on `LBManager`'s object hooks
(LBManager.h:313-393) and is optional.

Decision 2026-11-08 (review F12): a resumed coroutine runs inside the
`CkDynMgr.deposit` entry (or whichever entry resolved its future), so without more work
its time after the first suspension is charged to `CkDynMgr`, a group the balancer
cannot move. The binding therefore brackets every greenlet resume and suspend exactly
as Charm++ does for `[threaded]` entries (`Chare::CkAddThreadListeners`, ck.C ~171-198,
where a listener pops the chare from the callstack on suspend and pushes it on
resume): it calls `CkDynCallstackPush(target)` when it resumes a coroutine and
`CkDynCallstackPop()` when the coroutine suspends or finishes. This covers load and
GPU attribution for the whole coroutine, not only around kernel launches. The tracing
half is in 3.13. In scope for the step 4a follow-ups, because the `appspector` tests
read these load numbers.

### 3.13 Tracing

Projections attributes time to `_entryTable[ep]->name` (trace-projections reads the
entry index from the envelope and prints the registered name in the `.sts` file). The
C++ classes are shared by every Python class of a kind, but the registration is not:
`CkDynRegisterClass` calls `CkRegisterChare` once per Python class, so each Python
class has its own chare index and `ndims`, and `CkRegisterEp` once per method, so each
method has its own entry index. The `.sts` writer (trace-common.C:303-306) prints one
`CHARE` line per chare index with its name and dimension count and one `ENTRY CHARE`
line per entry with its name and chare index, and trace-projections uses the chare's
`ndims` to record the array index of the executing element (trace-projections.C:706).
Projections therefore sees Python classes, methods and element indices exactly as it
sees charmxi-generated ones, with no extra work, which today's design achieves only
partially (bare method names, and inline entries traced as `dummy_thread_ep` until
charm #4108). The trace modules are created by `_createTraces`, which for the shared
library is the weak default definition of 3.15 item 2 (today init.C:94-108 under
`CMK_CHARM4PY`): with `CMK_TRACE_ENABLED` it creates projections and summary, and the
usual `+traceroot`, `+logsize` and `+traceoff` options apply. Wheels ship a libcharm
built with `--enable-tracing` so that Projections works from a `pip install`; the
cost of the compiled-in trace checks on the plain-message number is measured at step
4a against a build without the option (the step 0 build has `CMK_TRACE_ENABLED 0`). User events keep the three wrappers
(`CkTraceRegisterUserEvent`, `CkTraceBegin/EndUserBracketEvent`, ck.C:2749-2762)
renamed `CkDynTrace*`; the trace modules are registered at start-up for the shared
library as init.C:1608-1611 does today, under the same generalized hook as the LB
modules (6.1). Coroutines: a suspended coroutine's time is not an entry-method
bracket; CkDyn offers `CkDynTraceBeginEntry(ep)` / `EndEntry` so the binding can
bracket the resumed portion of a coroutine with the same ep name
(`_TRACE_BEGIN_EXECUTE_DETAILED`), which is what `appspector` would need for its
per-entry loads (charm4py #311) without the Python timers it uses today.

Decision 2026-11-08 (review F13): the claim above that Projections sees element
indices exactly as for charmxi classes is not yet true, and three changes make it so.
(1) trace-projections takes the dimension count from the executing element's actual
`CkArrayIndex` when the registered class has `ndims` 0 (trace-projections.C:706-716
records the index only for `ndims >= 1`, and the binding registers arrays with
`ndims` 0). (2) The many-element broadcast delivery path (`deliver_many`, 3.6) emits a
begin and end execute event per element when tracing is on, so an array broadcast is
not one array-manager event per PE. (3) The Projections gate checks that an array
element index appears in the log, not only the names in the `.sts` file. Implemented 2026-11-08 on `ckdyn-dev` (6bac2f2b1): CkDyn records the dimension count of the first index a process sees for a class registered with ndims 0 into the chare table, which the .sts writer and the log writer read; trace-projections and the record layout are unchanged. Limits: a class used with mixed dimension counts keeps the first count, and the .sts line reports 0 if PE 0's process never held an element of the class.

Decision 2026-11-08 (review F12): coroutine tracing follows Charm++'s trace listener
for `[threaded]` entries (`Chare::CkAddThreadListeners`, ck.C ~171-198), which ends the
execute bracket when the thread suspends and begins a new one, tagged with the thread
id, when it resumes. The binding uses a `CkDynTraceBeginEntry` variant that takes a
thread id for each resume and `CkDynTraceEndEntry` for each suspend, together with
`CkDynCallstackPush/Pop` (3.12). Because the resume runs inside the
`CkDynMgr.deposit` entry, that entry is marked trace-disabled, so only the coroutine's
bracket is open and no execute bracket is nested inside another. In scope for the
step 4a follow-ups.

### 3.14 Zero-copy: host and device buffers

```c
typedef struct CkDynDevDesc {
  const void    *ptr;  uint64_t len;  uint8_t elem;  uint8_t ndim;  uint16_t tag;
  const int64_t *shape;
  int            device;     /* device ordinal */
  uint64_t       stream;     /* cudaStream_t / hipStream_t handle the data is valid on */
} CkDynDevDesc;

/* receiver-posted buffers (the nocopypost / nocopydevice forms) */
void CkDynPostBuffer(const CkDynTarget *me, int ep, int slot, const CkDynBufDesc*);
void CkDynPostDeviceBuffer(const CkDynTarget *me, int ep, int slot, const CkDynDevDesc*);
/* device completion */
void CkDynHapiAddCallback(uint64_t stream, const CkDynCallback *cb);
int  CkDynHasDeviceSupport(void);      /* 1 in a cuda/hip build */
```

Zero-copy slots exist only in `CkDynMsg` (3.4): a `CkDynMcastMsg` cannot carry them,
and a broadcast with an `NCPY` slot is sent as a copy (the reconverse broadcast
forwarding for zero-copy is a stub anyway, 8.4).

**Host (`NCPY` slot).** The sender passes a `CkDynBufDesc` and a completion function.
CkDyn constructs a `CkNcpyBuffer(ptr, len, cb, CK_BUFFER_UNREG)` (ckrdma.h:136-210;
the registration mode is `UNREG` by default so that no memory registration outlives
the message; `REG` is an option for bindings that pin buffers) and pups it into the
descriptor prefix of `buf` in the layout the generated marshalling code uses for
`nocopy` parameters (`impl_num_rdma_fields`, root node, then the descriptors;
xi-Parameter.C:388-396), and marks the envelope `CMK_ZC_P2P_SEND_MSG`. On the receiver,
`_processHandler` calls `CkRdmaIssueRgets` before the entry runs (ck.C:1399-1415);
the destination buffer is allocated by CkDyn (the view then reports a completed `NCPY`
slot whose pointer is that buffer, owned by the message and released with it) or is a
buffer the binding posted in advance with `CkDynPostBuffer`, which is
`CkMatchBuffer`/`CkPostBuffer` (ckrdma.h:522-595). The sender's completion function
receives `done_arg` when the source may be released; the binding uses it to drop its
reference to the numpy array. Rule from charm_best_practices.md "Zero-copy post API":
the data entry method can run inside the post call, so the binding records everything
it needs before posting.

Reconverse implements `CmiIssueRget` (conv-rdma.cpp:621) with a copy-based
request/response on fabrics without RDMA (macOS loopback) and with LCI gets elsewhere;
nothing in CkDyn depends on which.

**Device (`DEVICE` slot).** For an object with `__cuda_array_interface__` the binding
passes a `CkDynDevDesc`; CkDyn builds a `CkDeviceBuffer` (ckrdmadevice.h:77-140) and
marks the message `CMK_ZC_DEVICE_MSG` as the generated code does for `nocopydevice`
(xi-Parameter.C:406-408). Receive side: the binding either posts a device buffer
(`CkDynPostDeviceBuffer`) or receives one that CkDyn allocates through HAPI's pool,
exposed to Python as `_ckdyn.DeviceBuffer`, an object implementing
`__cuda_array_interface__` whose deallocator frees it. `CkDynHapiAddCallback`
replaces `CkHapiAddCallback` (ck.C:2736-2747) and takes a `CkDynCallback`, so the
completion is a normal future deposit with quiescence accounting (`hapiAddCallback`
ends in `QdCreate(1)`, charm_best_practices.md "A complete on enqueue device
receive"). Builds without CUDA/HIP compile these functions to abort with a message, as
today. The runtime requires `cuda` or `hip` in the triplet (6.4); the reconverse
device path exists in the tree (ckrdmadevice.C:61-994) but has not been run on this
Mac, which has no GPU; it is step 3 of the plan.

### 3.15 Disposition of the existing Ext API

Every symbol under `CMK_CHARM4PY` (surveys/survey-ext-api.md Tables A to C):

| today | disposition | replaced by |
|---|---|---|
| `StartCharmExt`, `registerCkRegisterMainModuleCallback` | generalized | `CkDynStart`, `hooks.register_classes` |
| `CkMyPeHook`, `CkNumPesHook` | deleted | `CkMyPe()`/`CkNumPes()` are macros over functions in reconverse; nanobind calls them |
| `CkRegister{MainChare,Group,SectionManager,ArrayMap,Array}Ext` | generalized | `CkDynRegisterClass` |
| `CkRegisterReadonlyExt`, `ReadOnlyExt` | generalized | the `ckdyn_ro` readonly with pack/unpack hooks |
| `CkCreateGroupExt`, `CkCreateArrayExt`, `CkInsertArrayExt`, `CkArrayDoneInsertingExt`, `CkMigrateExt` | generalized | `CkDynCreate*`, `CkDynArrayInsert`, `CkDynArrayDoneInserting`, `CkDynMigrateMe` |
| `Ck{Chare,Group,Array}ExtSend[_multi]` | generalized | `CkDynSend*` with `CkDynMsg` |
| `CkForwardMulticastMsg`, `SectionManagerExt` | deleted | CkMulticast (3.9) |
| `CkGroupGetReductionNumber`, `CkArrayGetReductionNumber` | generalized | `CkDynGetReductionNumber(target)` (3.7) |
| `CkSetMigratable` | kept, renamed | `CkDynSetMigratable` |
| `CkStartQDExt_*` (4) | generalized | `CkDynStartQD` |
| `CkExtContributeTo*` (4), `CkExtContributeInfo`, `extContributorType` | generalized | `CkDynContribute[Section]` |
| `CkReductionTypesExt`, `charm_reducers` | deleted | `CkDynReducers` |
| `external_py` reducer, `PyReductionExt` | generalized | `CkDyn::foreign` reducer with `hooks.reduce` |
| `CreateCallbackMsgExt`, `CkCallbackSendExt`, `isExtCallback`, Ext `CkCallback` ctors, charm4py `s_section` | deleted | message-kind conversion in `callEntry` (3.7); the existing `refnum` |
| `register*MsgRecvExtCallback` (chare, group, array, bcast), `MainchareCtorExt`, `ReadOnlyRecvExt`, `ArrayElemLeave/JoinExt`, `ArrayResumeFromSyncExt`, `ArrayMapProcNumExt` | generalized | the hook table (3.3) |
| `CkArrayBroadcaster::deliverAndUpdate` (charm4py overload) | generalized | selected by `ChareInfo::isDynamic`, calls `deliver_many` |
| `CkHapiAddCallback` | generalized | `CkDynHapiAddCallback` with a `CkCallback` |
| `CkTrace*UserEvent` wrappers | kept, renamed | `CkDynTrace*` |
| `_registerTreeLB`, trace registration in `_initCharm` (init.C:1598-1614) | generalized | a module-registration list run when no charmc moduleinit is linked (6.1) |
| late registration (register.C:48, 213) | generalized | `CkDynAllowLateRegistration` |
| `CcsRegisterHandlerExt` and the charm4py CCS layout (reconverse conv-ccs.cpp:36-57, 101-106) | generalized | `CkDynCcsRegister(name, fn)`: the handler receives `(const char *name, const void *data, int len)`; the 32-byte name field layout goes. This part is a reconverse pull request, not a charm one |
| `DEVICE_RECV_TYPE_CHARM4PY` (cmirdmautils.h:23; reconverse include/cmirdmautils.h:70) | deleted | unused; one charm and one reconverse change |
| `LBTurnInstrumentOn/Off`, topology and `Ccd*` functions | kept | general API, called directly |

Core changes outside `ckdyn.*` that remain after the Ext blocks are gone, all small and
general:

1. `ChareInfo::isDynamic` and the `deliver_many` branch in `CkArrayBroadcaster` (3.6).
2. `_initCharm`: weak `CkRegisterMainModule` and `_registerExternalModules`
   (init.C:1621-1644) and a registration list for library modules when no moduleinit
   exists (6.1). `_createTraces` gets the same treatment: init.C:94-108 defines it
   under `CMK_CHARM4PY` because no charmc-generated moduleinit exists for the shared
   library, and that definition makes a `charm4py`-target build unable to link an
   ordinary Charm++ program (duplicate symbol `_createTraces`, found in step 0,
   BENCHMARKS.md deviation 3). With a weak default definition in init.C, one build
   serves the C++ test binding, the Python binding and plain Charm++ programs.
3. The `CMK_CHARM4PY` macro survives only for build-system matters (shared library,
   TLS model, 6.1) and is renamed `CMK_LIBCHARM_SHARED`.
4. `CkSendMsgInline` prepares the envelope unconditionally (ck.C:1833-1840), the fix
   charm #4108 made for the group variant.
5. The libcharm build sets `REFNUM_TYPE="unsigned int"` (3.7); a configuration, not
   a code change.

### 3.16 The C++ test binding

`tests/charm++/ckdyn/` contains a C++ implementation of the hook table (`cbinding.C`,
about 300 lines) in which a "class" is a C++ struct with a method table of function
pointers, and tests that register classes through `CkDynRegisterClass` and exercise:
ping-pong between two array elements (the per-message lower bound for 7.4), group
and nodegroup broadcast, array broadcast with `deliver_many`, built-in and foreign
reductions to every callback target kind, a section multicast and section reduction
(under `+randomized_msgq`), a composite section over an array and a group, migration
with `+balancer RotateLB`, checkpoint and `+restart`, QD, priorities and expedited
sends, an `NCPY` slot through the post and non-post paths, and (under CUDA) a `DEVICE`
slot. These run in `recharm` CI from `TEST_DIRS` (charm/AGENTS.md) and are the
acceptance test of Layer 2 independent of Python. The CI build uses the default
16-bit `REFNUM_TYPE`; the tests keep future ids below 65536 so they pass with either
width, and the 32-bit path is exercised by charm4py's own CI, which builds with the
option (6.2).

### 3.17 The complete C API of `ckdyn.h`

Everything a binding calls, collected. Types defined earlier: `CkDynKind`,
`CkDynEntryDesc`, `CkDynClassDesc` (3.2); `CkDynTarget`, `CkDynHooks`,
`CkDynMsgView` (3.3, 3.4); `CkDynHdr`, `CkDynSlot`, `CkDynBufDesc`, `CkDynBufDone`,
`CkDynMsgBuilder` (3.4); `CkDynSendOpts` (3.5); `CkDynArrayOpts` (3.6);
`CkDynCallback` (3.7); `CkDynReducerInfo`, `CkDynRedData` (3.8); `CkDynSectionID`
(3.9); `CkDynDevDesc` (3.14). `CkDynMsg` and `CkDynMcastMsg` are opaque to C.

```c
/* versions and start-up */
int  CkDynApiVersion(void);
int  CkDynRefnumBytes(void);                     /* sizeof(CMK_REFNUM_TYPE) of this libcharm */
int  CkDynHasDeviceSupport(void);
void CkDynInit(const CkDynHooks*);
void CkDynStart(int argc, char **argv);          /* never returns */
void CkDynCreateManagers(void);                  /* only for a C++ main chare that is not MainchareDyn */
/* registration */
int  CkDynRegisterClass(const CkDynClassDesc*, int *first_ep);
void CkDynAllowLateRegistration(void);
int  CkDynRegisterClassLate(const CkDynClassDesc*, int *first_ep);
const CkDynClassDesc *CkDynClassOf(int chare_idx);
/* messages: the builder and view functions of 3.4 */
/* sends (3.5) */
void CkDynSendChare(const CkDynTarget*, int ep, CkDynMsg*, const CkDynSendOpts*);
void CkDynSendGroup(int gid, int pe, int ep, CkDynMsg*, const CkDynSendOpts*);
void CkDynSendGroupMulti(int gid, int npes, const int *pes, int ep, CkDynMsg*, const CkDynSendOpts*);
void CkDynSendNodeGroup(int gid, int node, int ep, CkDynMsg*, const CkDynSendOpts*);
void CkDynSendArray(int aid, int ndims, const int *index, int ep, CkDynMsg*, const CkDynSendOpts*);
void CkDynSendSection(const CkDynSectionID*, int ep, CkDynMcastMsg*, const CkDynSendOpts*);
void CkDynSendTarget(const CkDynTarget*, int ep, CkDynMsg*, const CkDynSendOpts*);
/* creation (3.6) */
void CkDynCreateChare(int chare_idx, int on_pe, CkDynMsg *ctor, CkDynTarget *out);
int  CkDynCreateGroup(int chare_idx, CkDynMsg *ctor);
int  CkDynCreateNodeGroup(int chare_idx, CkDynMsg *ctor);
int  CkDynCreateArray(int chare_idx, CkDynMsg *ctor, const CkDynArrayOpts*);
void CkDynArrayInsert(int aid, int ndims, const int *index, int on_pe, CkDynMsg *ctor);
void CkDynArrayDoneInserting(int aid);
void CkDynArrayDestroyElement(int aid, int ndims, const int *index);
/* callbacks and futures (3.7) */
void CkDynFutureSet(int pe, uint32_t id, CkDynMsg *value);
void CkDynCallbackSend(const CkDynCallback*, CkDynMsg*);       /* send through a callback description */
int  CkDynGetReductionNumber(const CkDynTarget*);
/* reductions (3.8) */
int  CkDynReducers(const CkDynReducerInfo **out);
int  CkDynForeignReducerType(void);
void CkDynContribute(const CkDynTarget*, const CkDynRedData*, const CkDynCallback*);
/* sections (3.9) */
void CkDynSectionCreateArray(int aid, int nelems, int ndims, const int *indices, int bfactor, const CkDynCallback *done, CkDynSectionID *out);
void CkDynSectionCreateGroup(int gid, int npes, const int *pes, int bfactor, const CkDynCallback *done, CkDynSectionID *out);
void CkDynSectionCombine(int nparts, const CkDynSectionID *parts, const CkDynCallback *done, CkDynSectionID *out);
void CkDynSectionDelete(const CkDynSectionID*);
int  CkDynSectionRedNo(const CkDynTarget*, const CkDynSectionID*);
void CkDynContributeSection(const CkDynTarget*, const CkDynSectionID*, const CkDynRedData*, const CkDynCallback*);
/* checkpoint (3.10) */
void CkDynCheckpoint(const char *dirname, const CkDynCallback *done);   /* CkStartCheckpoint */
/* quiescence (3.11) */
void CkDynStartQD(const CkDynCallback*);
/* load balancing (3.12) */
void CkDynAtSync(const CkDynTarget*);
void CkDynSetMigratable(const CkDynTarget*, int);
void CkDynMigrateMe(const CkDynTarget*, int to_pe);
void CkDynLBInstrument(int on);
void CkDynSetObjLoad(const CkDynTarget*, double secs);
void CkDynCallstackPush(const CkDynTarget*);
void CkDynCallstackPop(const CkDynTarget*);
/* tracing (3.13) */
int  CkDynTraceRegisterUserEvent(const char *desc, int id);
void CkDynTraceBeginUserBracketEvent(int id);
void CkDynTraceEndUserBracketEvent(int id);
void CkDynTraceBeginEntry(int ep);
void CkDynTraceEndEntry(int ep);
/* zero-copy and devices (3.14) */
void CkDynPostBuffer(const CkDynTarget*, int ep, int slot, const CkDynBufDesc*);
void CkDynPostDeviceBuffer(const CkDynTarget*, int ep, int slot, const CkDynDevDesc*);
void CkDynHapiAddCallback(uint64_t stream, const CkDynCallback*);
/* runtime wrappers so that the binding includes no Charm++ header */
int    CkDynMyPe(void);  int CkDynNumPes(void);  int CkDynMyNode(void);  int CkDynNumNodes(void);  int CkDynMyRank(void);
int    CkDynNodeSize(int node); int CkDynNodeFirst(int node); int CkDynNodeOf(int pe);
int    CkDynIsRestarting(void); int CkDynHasLB(void);
void   CkDynExit(int code);                      /* realCkExit */
void   CkDynAbort(const char *msg);
void   CkDynPrintf(const char *text);            /* CmiPrintf("%s", text) */
double CkDynWallTimer(void);
int    CkDynNumPhysicalNodes(void); int CkDynPhysicalNodeID(int pe); int CkDynPhysicalRank(int pe);
int    CkDynNumPesOnPhysicalNode(int node); int CkDynGetPesOnPhysicalNode(int node, int *pes, int max);
int    CkDynTopoTreeEdges(int pe, int root, const int *pes, int npes, int bfactor, int *parent, int *children, int max); /* getPETopoTreeEdges */
void   CkDynCallAfter(void (*fn)(void *arg, double now), void *arg, double msecs);   /* CcdCallFnAfter */
/* CCS */
typedef void (*CkDynCcsHandler)(void *global, const char *name, const void *data, int len);
void   CkDynCcsRegister(const char *name, CkDynCcsHandler);
int    CkDynCcsIsRemoteRequest(void);
void   CkDynCcsSendReply(int len, const void *data);
void  *CkDynCcsDelayReply(void);
void   CkDynCcsSendDelayedReply(void *token, int len, const void *data);
```

## 4. Layer 3a: the Python binding, `charm4py._ckdyn` (nanobind)

One extension module, built with nanobind (2.2 or later, the first line with
free-threaded CPython support), source `charm4py/src/ckdyn_py.cpp` plus
`objstore_py.cpp` (the port of `c_object_store.pyx`, 5.1). It includes `ckdyn.h`,
`numpy` headers for `PyArray_*` only where the buffer protocol is not enough, and
`greenlet/greenlet.h` for the C API of greenlets. It links `libcharm` as the Cython
module does today (surveys/survey-build.md section 6). Everything it exposes to Python
is a small, flat API that `charm.py` and `chare.py` call; users never import it.

### 4.1 Module state

Process-wide, written during registration and read-only afterwards:

```c++
struct MethodEntry {           // one per Charm++ entry index
  PyObject *func;              // the unbound Python function (strong ref)
  PyObject *name;              // "Class.method" (for errors and profiling)
  uint32_t  flags;             // CKDYN_EP_* (THREADED means run on a coroutine)
  PyObject *when;              // the parsed @when condition object, or NULL
  int       nargs;             // positional parameter count (for kwargs mapping)
  PyObject *argnames;          // tuple of names (kwargs mapping, chare.py:321-336 today)
  PyObject *defaults;
};
std::vector<MethodEntry> methods;        // indexed by ep
std::vector<ClassEntry>  classes;        // indexed by CkDyn chare index: the Python class, kind, ndims
```

Per PE thread (a `thread_local` struct, created on first use by that thread):

```c++
struct PeState {
  PyGreenlet *main;                      // the greenlet that runs the Charm++ scheduler on this thread
  std::vector<PyGreenlet*> idle;         // coroutine pool, 4.5
  CkDynMsgBuilder *builder;              // reused for every send
  PyObject *charm;                       // the charm4py.charm object (for slow paths)
};
```

Every Python chare instance carries its `CkDynTarget` as the attribute
`_ckdyn_target` (a `_ckdyn.Target` object), plus `_numthreads` (tests read it) and
`_nsuspended` (the count of its suspended coroutines, 4.5). Decided in step 4a
(SPEC-4a.md): chares stay plain Python classes rather than inheriting a nanobind
base type, because today's code creates chares with `object.__new__` and unpickles
them through a no-argument `__new__`, which a nanobind base type does not allow; an
attribute read costs one dictionary lookup on the slow paths only. The handle CkDyn
stores is the `PyObject*` of the instance with one strong reference, released in
`destruct`.

### 4.2 Registration and the dispatch table

`charm.registerInCharm` (charm.py:513-526) becomes: for each class and kind, build a
`CkDynClassDesc` from the `EntryMethod` list (`charm.classEntryMethods`), with
`name = "Class.method"`, `flags` from the decorators (`_ck_coro` → THREADED; the new
`@entry(...)` attributes, 5.4), and `user_id = ep`'s position in `methods`. CkDyn
returns `first_ep`; the binding fills `methods[first_ep + i]`. `charm.entryMethods`
(a dict tests inspect) stays as a Python mirror. Proxy classes are still generated in
Python (`chare.py` `__getProxyClass__`), but each generated proxy method is a
`_ckdyn.ProxyMethod` object (a nanobind type with a vectorcall slot) created by
`_ckdyn.make_proxy_method(ep, kind)` instead of a Python closure
(chare.py:321-349, 457-500, 721-773). Calling it runs C++: read the proxy's ids from
its C slots, scan `kwargs` for the reserved keys (`ret`, `awaitable`, `is_ray`,
`priority`, `expedited`), map remaining kwargs onto positional parameters using
`argnames`/`defaults`, build the message (4.3), send.

### 4.3 Argument conversion

Sender, per positional argument, by exact type (`Py_TYPE(o) == &PyLong_Type` and so
on; subclasses take the slow path):

| Python type | slot | note |
|---|---|---|
| `None` | NONE | |
| `bool` | BOOL | before `int`, since `bool` is a subclass |
| `int` | I64 or U64; overflow → tail | `PyLong_AsLongLongAndOverflow` |
| `float` | F64 | |
| `bytes` | BYTES (≤ 4 KB inline) or BUFFER (larger; copied) | |
| `str` | BYTES (UTF-8) | |
| `numpy.ndarray`, non-object dtype, C- or F-contiguous | BUFFER (copy) below the zero-copy threshold, NCPY above it (default threshold 1 MB, `charm.options.zerocopy_threshold`) | dtype → `elem`; shape and order in the slot; non-contiguous arrays are made contiguous first (today they are rejected by `memoryview`, charm.py:471-475) |
| `array.array`, `memoryview`, any object with a contiguous buffer | BUFFER | via `PyObject_GetBuffer(PyBUF_SIMPLE)`; `array.array` typecode → `elem` |
| object with `__cuda_array_interface__` (cupy, torch, numba) | DEVICE | 3.14; only in a device-enabled build, else `TypeError` |
| `Future` | FUTURE | the binding's future type has C slots `(pe, id)` |
| a proxy | PROXY | ids from the proxy's C slots; section proxies → SECTION |
| anything else | pickled into the tail | one `pickle.dumps(tuple_of_rest, protocol=5, buffer_callback=cb)`; `cb` turns out-of-band `PickleBuffer`s into BUFFER slots with `bflags` bit 2 set, so a numpy array nested in a list or dict is copied once into the message and once more out of it on receive, which is also what happens today through pickle (today's top-level `dcopy` path and this one differ only in that the slot carries the dtype and shape) |

The receive side inverts this, using the slot's `tag` to rebuild the sending type
(`str` from a BYTES slot tagged STR, `array.array` with its typecode, `memoryview`,
`numpy.ndarray`). BUFFER and NCPY slots become `numpy.ndarray` views over
the message (`PyArray_NewFromDescr` with a base object that holds the retained message,
3.4), writable for a point-to-point delivery, read-only for a broadcast or multicast
delivery where several elements share the message. Today every received array is
copied (charm.py:1282-1285). A user who mutates a received broadcast array gets a
`ValueError: assignment destination is read-only` where today they silently mutated a
private copy; 5.4 item 8 lists this change and `charm.options.copy_received_arrays = True`
restores the old behavior globally. `array.array` arguments come back as
`array.array` (typecode preserved), `bytes` as `bytes`. The tail is unpickled with
`pickle.loads(tail, buffers=...)` where the out-of-band buffers are the BUFFER slots
with `bflags` bit 2 set, in slot order. Objects that expose `__array_interface__`
without the buffer protocol are rare (every numpy array supports both); the binding
passes them through `numpy.asarray` first.

Return values and `ret=True`: when `CKDYN_MSG_BLOCK` is set, the method's return value
is sent to the FUTURE slot's target through `CkDynFutureSet`, converted by the same
table (so a numpy return travels as a buffer, not a pickle). `bcastret` gathers through
the foreign `gather` reducer as today (entry_method.py:63-74).

### 4.4 The deliver hook, the GIL and exceptions

```c++
static void py_deliver(void *bd, void *handle, int ep, uint32_t user_id, uint32_t ep_flags,
                       const CkDynMsgView *view) {
  nb::gil_scoped_acquire gil;                     // PyGILState_Ensure: a thread-state lookup and a counter; cheap, not free
  MethodEntry &m = methods[ep];
  PyObject *self = (PyObject*)handle;
  if (m.when || (ep_flags & CKDYN_EP_THREADED) || (view->hdr->flags & SLOW_FLAGS))
    return deliver_slow(m, self, view, ep_flags);  // when-conditions, coroutines, ret/awaitable/creation/ray
  ArgStack args(view);                             // converts slots into a small on-stack PyObject* array
  PyObject *r = PyObject_Vectorcall(m.func, args.with_self(self), args.n + 1, nullptr);
  if (!r) return handle_exception(m, self, view);
  Py_DECREF(r);
}
```

Rules:

- The GIL is released by `_ckdyn.start()` around `CkDynStart` and taken in every hook.
  Every `_ckdyn` call made from inside a Python method runs Charm++ code with the GIL
  held. Decision 2026-11-08 (review F14): an earlier version of this rule said such
  calls never deadlock against another PE thread; that is wrong for a process with
  more than one PE. PE A, in Python, can block on a runtime lock (for example
  `CkNodeReductionMgr`'s `lockEverything`, held while the reducer, which is the foreign
  reduce hook, runs, ckreduction.C:2155-2158, 2198-2205) while PE B holds that lock
  inside a hook and waits for the GIL. The rule until step 7 is one PE per process:
  `_ckdyn.start` refuses `CmiMyNodeSize() > 1`. Step 7 chooses among the options in
  8.3.
  `inline` entries and `CkMulticastMgr`'s local delivery can run a method nested inside
  another on the same thread (3.5; charm_best_practices.md "Zero-copy post API");
  `py_deliver` is re-entrant and the greenlet pool handles nesting (4.5).
- A Python exception never crosses back into Charm++. `handle_exception` implements
  charm.py:233-275 (`handleGeneralError`, `process_em_exc`): `SystemExit` → `CkExit`
  with the code (the whole test suite ends with the builtin `exit()`,
  surveys/survey-tests.md section 1); with BLOCK, the exception object is given the
  `remote_stacktrace` attribute and sent to the future (or contributed through the
  `_bcast_exc_reducer`/`gather` reducer for broadcasts); in interactive mode it is
  forwarded to PE 0; otherwise the traceback is printed and `CkAbort` is called. The
  slow path is Python code in `charm.py`, called from C++, so the behavior stays
  identical and the fast path pays nothing.
- A nanobind exception translator converts `CkDynError` (raised when CkDyn calls
  `hooks.error`) to `Charm4PyError`.

### 4.5 Coroutines without a greenlet per invocation

Today `_run_th` creates a greenlet per call (entry_method.py:95-102), and the
measured difference between the plain and coroutine ping-pong is 3.9 us per message
(BENCHMARKS.md: 6.14 against 2.27). The step 0 microbenchmarks on this Mac put a
greenlet created per call, run to completion, at 2.7 us, and a switch into and back
out of an existing greenlet at 0.37 us, so the pool below removes about 2.3 us of the
3.9 and the rest is coroutine bookkeeping in `threads.py` that the binding absorbs.

Design: each PE thread keeps a pool of idle *worker* greenlets whose run function is
the C++ `worker_main`. As built in step 4a (SPEC-4a.md 2.6), C++ owns the pool and the
switches into and out of a worker (`_ckdyn.spawn`, `_ckdyn.resume`), while pausing
and the parent re-parenting rule stay in `threads.py`; the measured cost is 1.48 us
per coroutine message, accepted as meeting the 1.4 us target (decision below); the
remaining Python frames (`_run_th`, `_run_coro`, `_run`) stay in Python:
(With the F12 load and trace brackets added 2026-11-08 the figure is 1.58 us; an A/B build
attributes 0.04 us to the two C calls and about 0.06 us to the binding's per-worker
record bookkeeping, a later optimization candidate.)

```c++
// Exposed to Python through nanobind as the run callable of every worker greenlet:
// PyGreenlet_New(run, parent) needs a Python callable, and a nb::cpp_function is one.
static nb::object worker_main(nb::object first_task) {
  PeState &pe = pe_state();
  PyObject *task = first_task.release().ptr();
  for (;;) {
    run_task(task);                                        // call the method; send its return value or exception
    pe.idle.push_back(PyGreenlet_GetCurrent());            // back in the pool
    task = PyGreenlet_Switch(pe.main, nullptr, nullptr);   // yield to the scheduler; the next task arrives as the switch value
    if (task == nullptr) {                                 // NULL means an exception was thrown into this greenlet
      if (PyErr_ExceptionMatches(PyExc_GreenletExit)) { PyErr_Clear(); break; }   // pool shrink or interpreter exit
      throw nb::python_error();
    }
  }
  return nb::none();
}
```

Decision 2026-11-08 (review F8): the measured 1.48 us at one PE, with spawn and resume
in C++ and the pause path in Python, is accepted as meeting the 1.4 us target, and the
performance side of the step 4a gate is closed. Moving the pause path into C++ is not
planned; it would be revisited only after step 7, if the GIL work changes this path
anyway. The `worker_main` above is the fully C++-driven form the target was derived
from.

The per-task attributes that today live on the greenlet object (`gr.obj`, `gr.notify`,
`gr.em_callstack`, entry_method.py:95-99, 107-109) live in the task record, which the
worker resets for each task, so a pooled greenlet carries nothing over.
`deliver_slow` for a THREADED entry pops an idle worker (or creates one, parent =
`pe.main`), and switches to it with the task `(self, method, args, flags, future)`.
If the method finishes without blocking, the worker returns to the pool and to main in
one switch pair. If it blocks (`Future.get()`, `Channel.recv()`, `self.wait(...)`,
`charm.iwait`), it calls `_ckdyn.pause()`, which is `PyGreenlet_Switch(pe.main)`; the
resumer (`Future.deposit` → `_ckdyn.resume(gr, value)`) switches back to the worker,
which returns from `pause()` with the value. The worker stays out of the pool while
blocked. This keeps `threads.py`'s model (one greenlet per blocked coroutine, a main
greenlet that runs the scheduler) and removes creation and teardown from the
non-blocking case, which is the common case for `@coro` methods that only
occasionally block. `obj._numthreads` is maintained as today for the two tests that
read it. Nested delivery (an inline entry delivered while a worker runs) starts a
second worker whose parent is the current greenlet, exactly the case
`threads.py:278-287` handles by re-parenting to main; the C++ code does the same.

Profiling (`charm.options.profiling`) keeps the per-entry timers of
`entry_method.py:127-157`, implemented in C++ around the vectorcall and the switches so
that it measures the same quantities with less overhead.

Decision 2026-11-08 (review F12): every greenlet resume and suspend in the binding is
bracketed as Charm++ brackets a `[threaded]` entry: `CkDynCallstackPush` of the
coroutine's object and a thread-id trace begin on resume, the matching pop and trace
end on suspend (3.12, 3.13).

**Suspended coroutines, migration and checkpoint.** A suspended coroutine is a
greenlet holding a slice of the C stack and Python frame objects whose pointers are
valid only in this interpreter; nothing can move it to another process (and
isomalloc-style stack migration, which Charm++ uses for `[threaded]` C++ entries,
neither covers interpreter state nor works across processes on macOS, machines/mac.md
2026-11-05). Today's rule is kept, "an element with active threads cannot migrate"
(threads.py:19-27, 258-259), but it is enforced so that a load balancer never trips
over it:

1. The binding keeps `obj._numthreads`, the count of coroutines of this element that
   are suspended (on a future, a channel, `charm.wait`, `awaitCreation`, a blocking
   `get`). When it goes from 0 to 1 the binding calls `_ckdyn.set_migratable(obj,
   False)` if the user left the element migratable, and restores `True` when it
   returns to 0. `CkMigratable::setMigratable` (ckmigratable.h:99) clears the
   migratable bit in the LB database record, so every balancer excludes the element
   from its decisions while a coroutine is suspended. Two cheap calls per transition,
   none for coroutines that never block.
2. An explicit `obj.migrate(pe)` with `_numthreads > 0` raises `Charm4PyError` naming
   the suspended methods, before anything is sent.
3. If an element still reaches `pup` with reason MIGRATE and `_numthreads > 0` (a
   coroutine that suspended between `AtSync` and the balancer's migrate call, the
   window Charm++ leaves open for C++ too), `pack_size` returns `CKDYN_PACK_REFUSED`
   and the program aborts with the element index and the method names (3.10). The
   alternative, silently dropping the coroutine, would lose the entry method's
   remaining work and any result it owed.
4. With reason CHECKPOINT the pack proceeds regardless: a checkpoint is a snapshot
   and the original keeps running, so the suspended coroutines live on in it. On a
   `+restart` (or a shrink/expand relaunch) they do not exist, exactly as suspended
   `[threaded]` C++ entry methods and the main chare's thread do not exist after a
   C++ restart: execution resumes through the checkpoint callback, or through
   `ResumeFromSync` after shrink/expand. Since every charm4py main chare constructor
   is a coroutine (charm.py:567-571) and typically blocks on a future, a program that
   uses checkpoint or shrink/expand must put its resume logic in entry methods or
   `resumeFromSync`, not after a `get()` in `main`; the documentation says so, and
   `tests/checkpoint/restart.py` (step 4c) is written that way.
   Decision 2026-11-08 (review F11): "the checkpoint callback" is the `resume` target
   of `charm.checkpoint(dir, resume)`, an entry method of a group, an array element
   or the main chare (3.7, 3.10), not a future. On the original run the call can also
   deposit to an optional future so that `main` continues; after `+restart` only
   `resume` runs, and `tests/checkpoint/restart.py` exercises it.
5. Futures are PE-pinned, as today (the result goes to the creating PE's table,
   4.6). An unresolved future held in a migrating element's state keeps working after
   the move because its `(pe, id)` names the creating PE, not the holder, and `get()`
   on it from the new PE sends the request to that table (4.6; today `get()` must run
   where the future was created, which the move would violate silently). After a
   restart, future ids from the checkpointed run are stale: `CkDynMgr`'s pup saves the
   per-PE id counter so new ids never collide with old ones, and a deposit or `get()`
   for an id the table does not know raises `Charm4PyError("future from before the
   restart")` on the depositing side instead of waiting forever.
6. `when` conditions and the messages buffered behind them are data (wait.py:89-93,
   152-156) and migrate and checkpoint as part of the element's state, as today.
   Channels keep today's rule: migration waits until channel establishment is
   complete (charm.py:746-747).
7. Cooperative coroutine migration is possible at an explicit call site, and exists
   as unmerged work: Zane Fink's charm4py branch `migreenleet` (commits e3fb530,
   7ac4229, 65a1c68, February-March 2025) adds `Chare.migrateAndWait(pe)`, which
   serializes the calling coroutine's own frame with the `sauerkraut` package
   (`copy_frame` with locals, excluding the chare itself), stores it in the chare's
   state, migrates, and on arrival re-creates the frame in a fresh greenlet
   (`deserialize_frame(run=True)`) so execution continues after the call
   (tests/migration/coro_migration.py on that branch). It moves one frame at a known
   point, not a coroutine blocked on a future deeper in a call stack, and it needs
   the chare's state to be the only thing the frame refers to. It fits the design as
   an extension: the binding's pack with reason MIGRATE includes the serialized frame
   when the element is suspended at `migrateAndWait`, and `unpack` starts the
   greenlet; items 1-3 stay the rule for every other suspension point. Listed in
   step 7; whether `sauerkraut` is maintained for Python 3.13 and 3.14 has to be
   checked first.

### 4.6 Futures

`threads.Future` keeps its Python class and semantics (deposit counting, `nvals`,
exceptions, `__getstate__` for the Ray store) but gets C slots `(pe, id)` and the id
becomes 32-bit (`FIDMAXVAL = 2**32 - 1`, threads.py:10, now safe because libcharm's
reference numbers are 32 bits wide, 3.7). Deposits from other PEs arrive through
`CkDynMgr[pe].deposit` (3.7) and reach `threadMgr.depositFuture` with no user-level
group entry in between (today: `CharmRemote._future_deposit_result`, charm.py:1185).
Collective futures (`allreduce`, chare.py:124-147) keep using the contributor's
reduction number as the shared id, through `CkDynGetReductionNumber`, without the
`% 65535` wrap (chare.py:130).

Two additions for migration and restart (4.5 item 5). A `get()` on a PE other than
the creating one (a future that moved with its element) creates a local future and
sends `(home id, local pe, local id)` to `CkDynMgr[home].forward`; the home table
answers with the value at once if it is deposited, otherwise records the forward and
sends the value when the deposit arrives. Today's store futures do the same for the
Ray API (`charm.get_future_value`, `_requested`), so the pattern is already in
threads.py. And `CkDynMgr::pup` saves the per-PE id counter (`lastfid`) so that ids
allocated after a `+restart` never collide with ids in checkpointed state; a deposit
or `forward` for an id the table has never allocated raises on the sending side.

### 4.7 The object store

`c_object_store.pyx` (279 lines, no Charm++ dependency) is ported to the same nanobind
module as `_ckdyn.ObjectStore` so that the package has no Cython build dependency.
Its interface (`MessageBuffer`, `ObjectStore` with `lookup_object`, `create_object`,
`request_location_object`, ...) is unchanged; the Ray API (`ray/api.py`,
`object_store.py`) does not change.

### 4.8 What `_ckdyn` exposes to Python

The whole surface `charm.py`, `chare.py`, `threads.py` and `reduction.py` use:

```python
# module functions
init(charm_obj, options) -> None                 # installs the hooks; records the charm object for slow paths
register_class(cls, kind, ndims, entries) -> int # entries: list of (func, name, flags); returns first ep; fills the method table
register_class_late(...)                         # interactive mode
start(argv) -> None                              # releases the GIL and runs CkDynStart; returns only at exit
my_pe() / num_pes() / my_node() / num_nodes() / wall_timer() / exit(code) / abort(msg) / printf(text)
physical-node and topology wrappers, one per CkDyn wrapper in 3.17
make_proxy_method(ep, kind) -> ProxyMethod       # a vectorcall object; called as pm(proxy, *args, **kwargs)
send(target, ep, args, flags, future) -> None    # the slow path behind ProxyMethod and internal sends
create_chare / create_group / create_nodegroup / create_array / array_insert / array_done_inserting / destroy_element
contribute(obj, data, reducer_type, callback) -> None
contribute_section(obj, sid, data, reducer_type, callback) -> None
reducers() -> list[(name, type, elem)]           # built at import from CkDynReducers
foreign_reducer_type() -> int
register_reducer(func, name) -> int              # the binding-side id used in message headers
section_create(proxy, indices_per_section, bfactor, done_future) -> list[SectionID]
section_combine(sids, done_future) -> SectionID
section_delete(sid) -> None
section_redno(obj, sid) -> int
reduction_number(obj) -> int
start_qd(callback) -> None
at_sync(obj) / set_migratable(obj, flag) / migrate(obj, pe) / lb_instrument(flag) / callstack_push(obj) / callstack_pop(obj)
checkpoint(dirname, future) -> None
pause() -> object                                 # switch to the PE's main greenlet; returns the value passed to resume
resume(greenlet, value) -> None
trace_register_user_event(desc, id) / trace_begin_user_bracket(id) / trace_end_user_bracket(id)
ccs_register(name, handler) / ccs_is_remote_request() / ccs_send_reply(bytes) / ccs_delay_reply() -> token / ccs_send_delayed_reply(token, bytes)
call_after(callable, msecs) -> None
hapi_add_callback(stream, future) -> None
has_device_support() -> bool
retain(view_owner) / release(...)                 # used only by the array base object, not by Python code

# types
class ChareBase: ...        # base of charm4py.Chare; C slots: target, handle state, _numthreads
class ProxyMethod: ...      # vectorcall; attributes .ep, .__name__, .__qualname__
class SectionID: ...        # 16-byte POD; hashable; picklable
class Callback: ...         # wraps CkDynCallback; built from a proxy method, a Future, or a SectionID + ep
class DeviceBuffer: ...     # __cuda_array_interface__ over a CkDyn-owned device allocation
class ObjectStore: ...      # 4.7
class MessageBuffer: ...    # 4.7
```

The API changes list in 5.4 is what a user sees; this list is what the Python package
sees, and it is the contract a Julia binding would reimplement over the same C API.

## 5. Layer 3b: the Python API layer

### 5.1 Module by module

| module (lines) | disposition | detail |
|---|---|---|
| `__init__.py` (40) | kept | exports unchanged; adds `NodeGroup`, `entry` |
| `charm.py` (1289) | rewritten in part, about 950 lines remain | deleted: `recvReadOnly/buildMainchare/recvChareMsg/recvGroupMsg/arrayMapProcNum/recvArrayMsg/recvArrayBcast/unpackMsg/packMsg` (277-491), `groupMsgBuf`, `arrayElemLeave` (740-756), the section half of `contribute` (771-789), `combine`/`split` bodies (791-850, replaced by 5.2), `startQD`'s four cases (852-865), `rebuild*` (1271-1285), `load_charm_library`'s interface switch (1229-1249; `+libcharm_interface cython` is still accepted and ignored because `auto_test.py` passes it). Kept: `Options`, `Charm4PyError`, `register`, `start` (the registration order rules), `registerAs/registerInCharm` (now building class descriptors), `CharmRemote` (minus `_future_deposit_result`), exceptions, `scheduleCallableAfter`, `iwait/wait`, profiling tables, topology, CCS, `printStats` |
| `chare.py` (926) | rewritten in part | `Chare.__init__`'s `_local` pool (36-65) deleted with `local_msg_optim`; `_getSectionLocations_` (179-225) replaced by a membership vote that returns indices (5.2); the proxy generators keep their structure but each method becomes a `_ckdyn.ProxyMethod` (4.2); `contribute`/`reduce`/`allreduce`/`AtSync`/`migrate`/`setMigratable` call `_ckdyn`; `Mainchare`, `Group`, `Array`, `ArrayMap` classes kept; `NodeGroup` added |
| `entry_method.py` (196) | kept, run paths move | `EntryMethod` keeps `name`, `epIdx`, `when_cond`, profiling fields; `_run*` (39-125) move to C++ (4.4, 4.5); `coro`, `coro_ext`, `when` decorators unchanged; `entry(**attrs)` decorator added |
| `threads.py` (370) | kept | `Future`, `CollectiveFuture`, `LocalFuture`, `LocalMultiFuture` unchanged in API; `EntryMethodThreadManager.pauseThread/resumeThread` call `_ckdyn.pause/resume`; `createFuture` uses 32-bit ids (4.6) |
| `wait.py` (269) | kept as-is | the `@when` AST machinery is pure Python; `test_when_syntax.py` imports `parse_cond_str` |
| `channel.py` (71), `pool.py` (481), `ray/` (150), `object_store.py` (83) | kept as-is | they use only proxies, futures and `charm.*` |
| `reduction.py` (351) | reduced to about 150 lines | `ReducerContainer` and the Python reducer functions kept; `ReductionManager.populateConversionTables` (201-300) deleted; `prepare` (303-351) becomes "pick a Charm++ reducer type from `_ckdyn.reducers()` by `(op, dtype)`, else foreign" |
| `sections.py` (201) | deleted | 3.9 |
| `charmlib/ccharm.pxd`, `charmlib_cython.pyx` (1299) | deleted | 4 |
| `c_object_store.pyx/.pxd` (342) | deleted, ported | 4.7 |
| `liveviz.py` (407), `appspector.py` (432), `interactive.py` (211), `loadfmt.py` (143) | kept | `appspector` reads `charm.groups` and `charm.entryMethods`, which stay as dicts; its per-entry timing may later use `CkDynTraceBegin/EndEntry` (3.13) |
| `charmrun/start.py` | kept | the reconverse launcher (`+pN` → N processes × 1 PE) is unchanged |
| `setup.py` (620) | replaced | by `CMakeLists.txt` + `pyproject.toml` with scikit-build-core (6.2) |

### 5.2 Sections and constrained groups in Python

`charm.split(proxy, numsections, section_func=None, elems=None, slicing=None)` keeps
its signature and return type (a list of section proxies). Implementation: a
broadcast of a reserved entry `__sectionVote__`, which the binding registers on every
class (it runs Python code, so it cannot be CkDyn's), to the collection, which runs
`section_func(self)` (or the slice test, or the `elems` membership test) on every
element and contributes `(index, section numbers)` through a `gather`-like foreign
reducer to a future on the caller; the caller then calls
`_ckdyn.section_create(proxy, indices_per_section, bfactor)` for each section, which
is `CkDynSectionCreateArray` or `CkDynSectionCreateGroup`, and wraps the returned ids
in section proxies. `charm.combine(*proxies)` calls `CkDynSectionCombine` on the
ids. Decision 2026-11-08 (review F17): the result is a deduplicated union, so an
element in two of the combined sections receives one copy of each multicast and
contributes once; it is its own CkMulticast section with its own cookie and reduction
number (3.9). It accepts only array sections whose arrays share an index type;
mixing group sections with array sections raises an error (5.4 item 15). Kale's
reservation: `combine` is a departure from Charm++, kept for
compatibility; in the long term either Charm++ adopts it or charm4py supports only
what Charm++ has. Section proxies keep `issec` and `section`; `section` becomes `(root_pe, sid_bytes)`
with the same tuple shape tests use (`proxy.section[1]`, sections/multirand-split-combine.py).
Group slicing `g[::2]` and array slicing `a[0:8:2, 1:8:2]` (chare.py:442-455, 702-719)
go through `split` with `elems`/`slicing` as today.

`Group(C, onPEs=[...])` creates the group with the PE list (3.6); the branches on
listed PEs get a `thisProxy` that is the group-section proxy CkDyn built, so broadcasts
and reductions on `thisProxy` reach only the listed PEs, which is the current contract
(sections/constrained_groups.py). The `cons` tuple protocol between `group_ckNew` and
`SectionManager.createGroupSectionDown` (chare.py:544-569, sections.py:57-99) goes.

Status 2026-11-08, step 4b implemented on `thin` (docs/design/SPEC-4b.md, which lists
the protocol and eight differences from this section, each kept). The main ones:
`split`, `combine`, slicing and `Group(C, onPEs=...)` must be called from a coroutine
and return when every member has joined (the `done` callbacks), so no member sees a
user message before `section_joined`; the vote entry is the reserved `Chare` method
`_sectionVote_`; in a constrained group `thisProxy` becomes the section proxy right
after the constructor, when the branch joins the section; a built-in reduction result
relayed to a section entry is recognized by its one BUFFER slot of `ndim` 0 with tag 0,
so the binding tags a 0-dimensional numpy argument (`TAG_ND0`).

### 5.3 Reductions in Python

`Reducer.sum/product/max/min/logical_*` stay tuples of `(op, python_fn)`;
`ReductionManager.prepare` maps `(op, dtype)` to a Charm++ reducer type from the table
`_ckdyn.reducers()` (name, type, elem), built once at import. Custom reducers
(`Reducer.addReducer`) get a per-process id in registration order; the contribution
is a CkDyn message with the data in the slots (so a numpy contribution to a custom
reducer is not pickled either) and the id in the header. The `hooks.reduce` callback
builds the list of Python values from the views, calls the Python reducer, converts
the result back. `gather` keeps its pre/post-process functions.

### 5.4 API changes, each with its reason

Additions (no existing program changes behavior):

1. `NodeGroup(C, args=[])`: one branch per process; `thisIndex` is the node number;
   entries may be marked `@entry(exclusive=True)`. Reason: missing feature; `NodeGroupDyn`
   needs no binding code beyond the class kind.
2. `@entry(expedited=False, inline=False, immediate=False, notrace=False,
   exclusive=False)` decorator on chare methods, and `priority=` / `expedited=`
   reserved keyword arguments on proxy calls. Reason: entry-method attributes and
   message priorities were not exposed; they are the Charm++ features the analysis
   listed as absent. `ret`, `awaitable` and `is_ray` were already reserved
   (chare.py:340, 476-483).
3. `charm.checkpoint(dirname, resume)` and `+restart <dir>`. Reason: 3.10 makes it
   work; today it cannot. Decision 2026-11-08 (review F11): `resume` is an entry method
   of a group, an array element or the main chare, invoked after `+restart`; on the
   original run the call can also deposit to an optional future so that a coroutine
   `main` continues (3.7, 3.10).
4. `charm.options.zerocopy_threshold` (bytes; default 1 MB) and
   `charm.options.copy_received_arrays` (default False). Reason: 4.3.
5. `charm.hapiAddCudaCallback` keeps its signature; device arrays as entry-method
   arguments become possible in device builds (3.14).

Removals and changes:

6. `charm.options.local_msg_optim` and `local_msg_buf_size` are accepted and ignored
   with a one-time warning. Reason: the local-message path stored Python objects by
   reference and "broke Charm++ semantics in some cases" (CHANGES.rst v1.1 note); it
   was already off by default; and it is the only part of today's design that bypasses
   the message. Tests `dcopy/test_dcopy.py`, `exceptions/pool.py` and
   `sections/multirand-split-combine.py` set these options and keep working.
7. `charm.options.pickle_protocol` default becomes 5 (was -1, the highest, which is 5
   on every supported Python). Reason: out-of-band buffers (4.3) need protocol 5;
   setting a lower protocol disables that path but still works.
8. Received numpy arrays from a broadcast or section multicast are read-only views;
   from point-to-point sends they are writable views; neither is a copy. Reason: 4.3.
   `copy_received_arrays=True` restores copies. The test survey's feature rows for
   dcopy and broadcasts show no test that mutates a received broadcast array, which
   is not a proof; the gate of step 4 is where it is checked.
9. `charm.getTopoSubtrees` is kept (tests/topo/topo_treeAPI2.py) but no longer used
   internally.
10. Migration of elements that belong to sections becomes allowed (today it aborts,
    charm.py:742-744). Reason: CkMulticast tracks members.
    Decision 2026-11-08 (review F17): step 2 adds a C++ test with section reductions
    across `RotateLB` migrations under `+randomized_msgq`, since this relies on
    CkMulticast's rebuild and on pupped cookies.
11. The `+libcharm_interface <name>` command-line option is accepted and ignored.
    Reason: `auto_test.py` passes it; only one interface exists.
12. Internals that tests touch (`self._numthreads`, `charm.groups`,
    `charm.entryMethods`, `charm4py.threads.Future`, `charm4py.wait.parse_cond_str`)
    are preserved with the same meaning. Reason: four tests read them
    (surveys/survey-tests.md "Tests that touch charm4py internals").
13. (Deferred past step 4a; see 8.6 item 1.) `Chare(C, args=[...], onPE=p)` creates a real singleton chare (`ChareDyn`, 3.6)
    instead of a one-element array; `onPE` omitted or -1 hands the chare to the seed
    balancer, as in Charm++. The proxy is a `CkChareID`-based proxy: it supports
    entry-method calls, `ret=True` futures and callbacks; `contribute` is not
    available on it (a singleton has no reduction tree, as in Charm++), and
    `migrate`, `AtSync` and `setMigratable` raise on it. `Chare(C, ..., migratable=True)` keeps today's one-element-array form
    with its full API, pending a Charm++-side design for migratable singletons
    (8.6 item 1). Reason: a one-element array costs per-PE state and a creation
    broadcast, which rules out programs that create billions of chares; the two
    library users of singletons (`PoolScheduler`, Ray actors) never migrate.
14. Migration of an element with a suspended coroutine stays an error, but load
    balancers no longer choose such an element: it is marked non-migratable while any
    of its coroutines is suspended (4.5). Checkpoint of such an element is allowed;
    the suspended coroutines are not part of the checkpoint, and restart resumes from
    the checkpoint callback or `resumeFromSync`. A future created before a restart
    raises when used after it instead of hanging. Reason: 4.5 items 3-5.
15. `charm.combine` accepts only array sections whose arrays share an index type, and
    raises `TypeError` for a mix of group and array sections or of arrays with different
    index types (tests/sections/multirand-split-combine.py:120-160 is adapted to combine
    its array sections only). Reason: decision 2026-11-08 (review F17); a combined
    section is a CkMulticast multi-array section, which covers exactly that case, and
    charm4py supports what Charm++ supports (3.9).

Everything else in the documented API (charm4py/docs/*.rst, the list in
surveys/survey-tests.md) is unchanged.

---

## 6. Build and packaging

### 6.1 The Charm++ side

Today's `charm4py` CMake target (CMakeLists.txt:529-537, 938-977) builds `libcharm`
as a shared library from the whole archives `ck`, `charm_cxx_utils` and `isomalloc`
plus `libreconverse` as a separate shared library, defines `CMK_CHARM4PY`, and leaves
out `-ftls-model=initial-exec` (detect-features-cxx.cmake:43-54). What changes:

1. `ckdyn.C` and `CkDyn.ci` join the ck-core sources in `src/ck-core/CMakeLists.txt:22-35` unconditionally, so
   every build has them and `tests/charm++/ckdyn` runs on every target.
2. `CMK_CHARM4PY` is renamed `CMK_LIBCHARM_SHARED` and keeps only its build meaning:
   the shared-library link, the TLS model, and the "no charmc moduleinit" start-up
   (3.2, 3.15). The reconverse definition (cmake/reconverse/CMakeLists.txt:97-101)
   goes with `CcsRegisterHandlerExt`.
3. The `init.C:1598-1614` block ("register TreeLB and the trace modules because there
   is no moduleinit") becomes a list, `_registerBuiltinModulesForSharedLib()`, that
   also registers CkMulticast's dependencies, the LB strategies that libcharm should
   offer (today only `TreeLB` and the strategies it loads at run time), and the trace
   modules under `TRACING`, with `CkRegisterMainModule` resolved as a weak symbol.
4. The target name `charm4py` stays as what `./build charm4py <triplet>` accepts, with
   `libcharm` as an alias, because setup.py, docs and CI all spell it that way.
5. Nothing is exported selectively (there is no version script today,
   surveys/survey-build.md section 5); the binding needs only `CkDyn*` symbols and the
   few `Cmi*`/`Ccs*` functions it calls, so a version script restricting exports to
   `Ck*`, `Cmi*`, `Ccs*`, `Ccd*`, `LB*` and `CkDyn*` is a later clean-up, not required.
6. The libcharm that wheels ship is built with `--enable-tracing` (3.13) and, once
   reconverse implements the Converse half (step 8), `--enable-shrinkexpand` (3.10).
   A source build may drop either.

Classic Converse: no changes. `classic` is frozen; CkDyn is not ported there. The old
charm4py keeps building against `classic` with the Ext API, which `classic` retains
forever. This is decision 8 of section 0 and risk 8.2.

### 6.2 The charm4py side

`pyproject.toml` switches the build backend to scikit-build-core with nanobind as a
build requirement; `setup.py` is replaced by a `CMakeLists.txt` that:

1. Finds or builds `libcharm`: if `CHARM4PY_CHARM_BUILD_DIR` has `lib/libcharm.*`, use
   it; else run `./build charm4py $CHARM4PY_BUILD_TRIPLET -j$N --with-production
   --disable-fortran --with-refnum-type="unsigned int"` in `CHARM4PY_CHARM_DIR` (the
   recipe setup.py:297-436 runs, with the LCI source-dir forwarding, plus the
   reference-number width of 3.7). The binding checks `CkDynRefnumBytes() == 4` at
   import so that a libcharm built without the option is refused with a clear message.
2. Builds `_ckdyn` with `nanobind_add_module`, include `<build>/include` (for
   `ckdyn.h`, `charm.h` is not needed), link `-lcharm`.
3. Copies `libcharm`, `libreconverse`, `liblci`, `liblct` into `charm4py/.libs` and
   sets the extension's rpath to it, as setup.py:146-294 does with
   `install_name_tool`/`patchelf`. In a wheel build this copying is unnecessary:
   `cibuildwheel`'s repair step (`auditwheel repair` on Linux, `delocate` on macOS)
   bundles every shared library the extension depends on and rewrites the rpaths, so
   the hand-written library renaming can go once wheels are the distribution path.
4. Writes the version file and the `reconverse` marker `charmrun/start.py` reads.

Run idiom unchanged: `python -m charmrun.start +p2 app.py` or
`lcrun -n 2 python app.py +pe 2` (charm4py/docs/install.rst:150-164).

### 6.3 Wheels and CI

- `build_wheels.yml` currently builds against `classic` only (surveys/survey-build.md
  section 8). Thin charm4py wheels are reconverse wheels: `cibuildwheel` with
  `manylinux_2_28` x86_64 and aarch64, and macOS arm64, each job building Charm++ with
  the reconverse triplet first (the reconverse CI job in `charm4py.yml` already does
  this on ubuntu-26.04 and macos-26). LCI's dependencies (libfabric or UCX) are
  dynamically linked into `liblci`; for a portable wheel LCI is built with its
  `tcp`/`ofi` backends against a libfabric that `auditwheel` bundles, and a note tells
  cluster users to build from source against the site's libfabric or UCX. This is the
  one packaging question with real uncertainty (8.5).
- The CI matrix keeps the reconverse job as the primary gate and adds a free-threaded
  interpreter (`3.14t`) once step 7 lands; the classic job disappears from this
  repository's branch (the old charm4py's CI stays on its own branch).

### 6.4 CUDA and HIP

`./build charm4py reconverse-linux-x86_64 cuda` (or `hip`) builds hybridAPI and links
`cudart hybridapi` into `libcharm` (CMakeLists.txt:958-964) and turns on reconverse's
and LCI's device support (survey-build.md section 3). The `_ckdyn` module is the same
source in both cases; it asks `CkDynHasDeviceSupport()` at import and raises a clear
error for device arguments otherwise, replacing the Cython compile-time
`HAVE_CUDA_BUILD` (setup.py:587-607, charmlib_cython.pyx:927-931). Device builds are
source builds (`pip install . ` with `CHARM_EXTRA_BUILD_OPTS=cuda`), not wheels: a
wheel cannot bundle the CUDA runtime, and the site's CUDA, libfabric and MPI/PMI all
need to match.

---

## 7. Migration plan

Every step ends with a system that builds and passes a named gate. Steps 1 to 3 are
Charm++ pull requests on top of `recharm` from the `ckdyn` branch; step 4 is a charm4py
branch; step 5 deletes the Ext API; 6 and 7 are packaging and extensions. The old
charm4py keeps working against the new libcharm through step 4 because the Ext API and
CkDyn coexist until step 5. Decision 2026-11-08 (Kale): the Ext API stays in `recharm`
until the thin charm4py has replaced the old one in use, not merely passed its tests;
step 5 is split accordingly (see its entry).

Gates refer to `charm4py/tests` paths and to the reverse index in
surveys/survey-tests.md section 4. "auto_test (N)" means `python auto_test.py` with
`CHARM4PY_TEST_NUM_PROCESSES=N` on reconverse, which runs every test at N processes
of one PE, or at the test's `force_min_processes` if larger.

### 7.1 Steps

**Step 0: baselines.** Extend `tests/benchmark/pingpong.py` with a payload option
(`-n BYTES`, a numpy array) and record, with the protocol of 7.4: plain and coroutine
ping-pong with an int, with a 1 MB array, and `tests/reductions/bench_reductions.py`,
on (a) today's charm4py on reconverse, one process, (b) two processes. Also time the
C++ ping-pong from `tests/charm++/ckdyn` once it exists (step 1) as the lower bound. Gate:
the numbers are in `BENCHMARKS.md` next to this file. No code changes.
Done 2026-11-07: BENCHMARKS.md holds the numbers summarized in 7.4; the `-n` option is
commit 50a67fe on charm4py branch `thin-bench`; the C++ floor came from
`benchmarks/charm++/pingpong` on a `charm++`-target build of the same source
(`charm/reconverse-darwin-arm8-cxx`), because the `charm4py`-target build cannot link
it (3.15 item 2).

**Step 1 (charm PR): CkDyn core.** `ckdyn.h/.C`, `CkDyn.ci`, `CkDynMsg`, registration,
hooks, `MainchareDyn`, `ChareDyn`, `GroupDyn`, `ArrayElemDyn`, `ArrayMapDyn`, sends with
options, `callEntry` conversion of reduction and system messages, the 32-bit
`REFNUM_TYPE` for the libcharm target, built-in reductions with the exported reducer table, `CkDynStartQD`, trace
names, the `ckdyn_ro` readonly, `CkDynMgr` with future deposit, the C++ test binding
with ping-pong, broadcast, reduction-to-every-target, QD, priority and expedited
tests, and `doc/internals/ckdyn.md`. The Ext API is untouched. Gate: `tests/charm++/ckdyn`
in `reconverse-ci.yaml` `TEST_DIRS`, both `+pe 2` in one process and `lcrun -n 2`;
the existing charm CI; today's charm4py auto_test (2) against this libcharm (proves
nothing regressed in the shared code: `ChareInfo::isDynamic`, the weak main-module
symbol).
Status 2026-11-07 20:50 PST: gate met. Branch `ckdyn` (commit 521a639bc, rebased on
recharm 16f27646b) is pull request charmplusplus/charm#4193; all ten CI checks pass;
`tests/charm++/ckdyn` passes on `+pe 1`, `+pe 4 +balancer RotateLB` and `lcrun -n 2`;
the `charm4py`-target libcharm rebuilt from the branch with `REFNUM_TYPE="unsigned
int"` runs today's charm4py test suite with no regression (80 of 86 entries pass; the
3 skips and 3 failures are missing numba, torch and matplotlib in the venv, identical
with the pre-branch libraries). Not yet exercised by the test: constrained groups,
array insertion, custom maps, callbacks, checkpoint. Deviations from the text above are
folded into 3.2, 3.4, 3.5, 3.6 and 3.17: group dependencies and priority bits are
given to `CkDynMsgBegin`, `CkDynSendOpts` is flags only, `CkDynCreateGroup` takes the
PE list as a slot plus a flag, `CkDynArrayInsert` takes the class from the
constructor message's ep, and the per-class `hooks` pointer of 8.7 exists from the
start. The reserved AtSync entry of the registration table is not registered:
`CkDynAtSync` calls `AtSync()` on the local element directly.

**Step 2 (charm PR): CkDyn collectives and lifecycle.** `NodeGroupDyn` with
exclusive entries, sections over CkMulticast with section reductions, the foreign
reducer, migration and checkpoint pup for every kind, LB hooks and `CkDynCallstack*`,
constrained groups, `deliver_many`. Tests added to the C++ binding: section multicast
and reduction under `+randomized_msgq` with several seeds, migration under
`+balancer RotateLB` (one process, since cross-process thread migration cannot run on
this Mac, machines/mac.md), `+restart` from a checkpoint, a nodegroup exclusive test
at `+pe 4`. Gate: the same as step 1 plus those tests.
Decision 2026-11-08 (review F17): the test list also gains section reductions across
`RotateLB` migrations under `+randomized_msgq`. Decision 2026-11-08 (review F15): the
ABI changes listed in 8.7 as due before `CKDYN_API_VERSION` 1 is frozen are made in
this step.

**Step 3 (charm PR): zero-copy.** `NCPY` slots through the post and non-post paths,
`DEVICE` slots, `CkDynHapiAddCallback`. C++ tests: a 16 MB `NCPY` ping-pong one and
two processes (copy-based on this Mac), the same on Anvil over the fabric, and a
`DEVICE` test built with `cuda` on Delta or Anvil GPU nodes. Gate: those tests; CPU
tests in CI, GPU tests run by hand and recorded.
Decision 2026-11-08 (review F18): when step 3 lands, the GPU test also gets a periodic
job on Delta, so that a regression after step 3 is caught.

**Step 4 (charm4py branch `thin`): the binding and the Python cut-over.** Done in
four sub-steps, each gated on a subset of auto_test so that the suite is green for the
subset and red only for the parts not yet ported:

| sub-step | work | gate (tests that must pass at auto_test 1, 2 and 4) |
|---|---|---|
| 4a | `_ckdyn` module; registration; proxies as `ProxyMethod`; sends and plain entries; coroutines with the pool; futures; built-in and custom reductions; callbacks; QD; exceptions; `exit()` handling; readonlies; topology; `scheduleCallableAfter` | `entry_methods/*`, `collections/*`, `futures/*`, `reductions/*` except `section_reduction`, `thread_entry_methods/*`, `when/*`, `callbacks/*`, `qd/qd`, `dcopy/test_dcopy`, `channels/*`, `exceptions/test`, `topo/*`, `array_maps/test1`, `charm_remote`; `benchmark/pingpong` at the targets of 7.4 |
| 4b | sections and constrained groups (5.2); delete `sections.py` | `sections/*`, `reductions/section_reduction`, `appspector/loads` |
| 4c | migration, `AtSync`, `setMigratable`, `migrated()`; checkpoint API | `migration/*` (with the LB args from `test_config.json`), a new `checkpoint/restart.py` |
| 4d | pool, Ray, interactive mode and remote exec, CCS, liveViz, appspector; delete `charmlib/`, port the object store | `pool/*`, `exceptions/pool`, `interactive/test1.in`, the examples in `test_config.json` including `ray/*`, `liveviz/test_liveviz_unit` |

Gate for the step as a whole: full auto_test at 1, 2 and 4 processes on this Mac and
in the reconverse CI job; `grep -r charmlib charm4py/` empty; no Cython in
`pyproject.toml`.

Status 2026-11-07 22:15 PST, step 4a: functional gate met on charm4py branch `thin`
(commit ed9ae2a, against charm branch `ckdyn` at the single-copy builder and the
spanningTree fix): every test in the 4a list passes at 1, 2 and 4 processes on this
Mac (`logs/gate-4a.md`; `test_different_coroutines` needs 2 PEs by construction).
The contract the two halves were written to is `SPEC-4a.md`; its decisions that
differ from sections 4 and 5 are recorded in 4.1, 4.5, 5.4 item 13 and 8.6 item 1.
pingpong, one-way, 1 PE: plain 0.38 us (target 1.0, was 2.27), coroutine 1.48 us
(target 1.4, was 6.14), 1 MB 24.9 us (target about 23, was 59.6); two processes over
TCP loopback: plain 24.5 us (was 26), coroutine 27 us (was 44). The two misses are
the remaining Python frames on the coroutine path (`_run_th`, `_run_coro`, `_run`,
two greenlet switches) and one memcpy plus the numpy view construction for 1 MB; the
independent review (`REVIEW-2026-11-07.md`, F8) expected the coroutine miss. Not in
4a: sections, constrained-group `thisProxy`, migration tests, pool, Ray, interactive
mode, and the single-process multi-PE configuration (`+pe 2` in one process
segfaults in about one run in eight inside the worker greenlet; that configuration
is step 7 and the GIL rule of 4.4 is not yet enforced, review F14).

Status 2026-11-08, step 4c: gate met on `thin` against `ckdyn-dev` c81fbeecc
(docs/design/SPEC-4c.md; logs/gate-4c). `migration/test_migrate` (GreedyRefineLB, and
RotateLB), `migration/test_nonmigratables`, `migration/chare_migration` and
`examples/particle` (GreedyRefineLB, and RotateLB) pass at `lcrun -n 2` and `-n 4`,
with every particle cell moved in every RotateLB period; new tests
`thin/atsync_in_coroutine` (F9: an element whose coroutine calls `AtSync` and returns
is migrated; an element with a suspended coroutine is skipped by the balancer and
`migrate()` raises; F10: 3000 `AtSync`/`resumeFromSync` iterations without a balancer
do not nest; `@when` queues survive the move) and `thin/sections_migrate` (section and
combined-section reductions across RotateLB moves, 5.4 item 10) pass at both counts;
`thin/checkpoint_restart` passes at one and two processes; the 4a and 4b gates show no
regression. Changes: the sizing-to-packing cache moved into `charm._pack`, which also
no longer removes the element from `charm.arrays` (`_destruct` does), so a sizing pass
alone (`AtSync` under `+MetaLB`) neither refuses nor changes state; a non-picklable
attribute aborts with the class and index named; `Chare.__init__` is called directly
before a user constructor. Found outside the binding: an element moved by
`migrate()` that then calls `AtSync` hangs the next balancing step at two processes,
also in plain C++ (SPEC-4c.md 6). The checkpoint test of the gate is
`thin/checkpoint_restart`, written in the 4a follow-ups, in place of the planned
`checkpoint/restart.py`. Long-lived received numpy views were measured (SPEC-4c.md 7):
no stall, memory grows by the retained messages.

**Step 5 (charm PR): delete the Ext API.** Remove every `CMK_CHARM4PY` block listed in
3.15, rename the macro, restore the C++ section callback, update `README.charm4py` and
the internals chapter. Gate: charm CI; the thin charm4py's auto_test against the
result; `grep -rn "Ext\b" src/ck-core` shows nothing charm4py-related.
Decision 2026-11-08 (Kale): split into 5a and 5b. **5a (with the Layer 2 merge):** make
CkDyn independent of `CMK_CHARM4PY`: the registration hook and late registration work
in every build (today they are under `#if CMK_CHARM4PY`), the macro keeps only its
build meaning (a shared libcharm with 32-bit refnums), and the Ext API remains, so one
libcharm serves both the old charm4py (through the Ext API) and the thin one (through
CkDyn); the old charm4py's test suite against that libcharm is part of the gate (80/86
on 2026-11-07, identical to the old libs). **5b (later, a separate PR):** delete the
Ext API only after the thin charm4py has replaced the old one for its users (a 2.0
release in use, the 1.x line retired). Until then the old charm4py stays alive as the
fallback.

**Step 6: packaging.** scikit-build-core build, reconverse wheels through
`cibuildwheel` on Linux x86_64/aarch64 and macOS arm64, a CUDA source-build recipe
checked on Delta, documentation (`docs/install.rst`, `gpus.rst`). Gate: `pip install
charm4py` from the built wheel on a clean virtualenv runs `examples/simple/hello_world.py`
and auto_test (2); the CI matrix from 6.3 is green.

**Step 7: extensions that the design enables.** In any order, each with its own test:
`NodeGroup` in the Python API; `@entry` attributes and priorities; zero-copy numpy
arguments above the threshold; device arrays as arguments (closes charm4py #309 along
with the reconverse device path); multi-PE processes on a free-threaded interpreter;
per-entry Projections traces for coroutines via `CkDynTraceBegin/EndEntry`;
cooperative coroutine migration at `migrateAndWait` from Zane Fink's `migreenleet`
branch (4.5 item 7). Projections itself is not an extension: step 4a's gate includes a run with `--enable-tracing`
libcharm and `+logsize` whose `.sts` file lists one `CHARE` line per Python chare class
and one `ENTRY CHARE` line per method with the `Class.method` name (3.13).
Decision 2026-11-08 (review F13): that gate also checks that an array element index
appears in the log. Decision 2026-11-08 (review F12): the coroutine trace and
callstack brackets move from this list to the step 4a follow-ups (3.12, 3.13).
Decision 2026-11-08 (review F14): multi-PE processes in this step choose among the
three options of 8.3, and the GIL-held-while-busy measurement of 8.3 is made here.

**Step 8 (reconverse PR, then charm PR, then charm4py): shrink/expand.** Reconverse
gains the Converse half that classic charmrun and netlrts provided (`Cmi_isOldProcess`,
`Cmi_myoldpe`, `ConverseCleanup`, relaunch of the job with the new process count under
the launcher, `+shrinkexpand_basedir`; init.C:226-347, 717-719 and 1489-1494 list what
ck-core expects); then `--enable-shrinkexpand` links on `recharm` and
`examples/charm++/shrink_expand` runs there, which is the gate for the runtime half.
The charm4py half is a Python port of that example (main chare with state, an array
with `AtSync`, `++server`, the CCS client from the example) that shrinks from 4 to 2
PEs and expands back in one process; gate: it passes on this Mac and on Anvil. CkDyn
needs no change for this step (3.10); it waits on nothing in steps 1-7 except 4c.
Decision 2026-11-08 (review F18): the charm4py gate above is replaced. Shrink/expand as
Charm++ defines it is a relaunch with a different process count, so the gate is a
checkpoint followed by a relaunch under lcrun with fewer processes and then with more
(shrink and expand), passing on this Mac and on Anvil; launcher support for the
relaunch is step 8 work. The plan follows Aditya's ongoing shrink/expand work in
Charm++ and builds on it where possible.

### 7.2 Dependencies and parallelism

Steps 1 to 3 are sequential in one branch but step 4a can start as soon as step 1 is
reviewed (4a needs only the core). 4b needs step 2; 4c needs step 2; 4d needs nothing
beyond 4a. Step 5 needs 4 complete. Step 0 and the C++ ping-pong number are the first
week's work and need nobody's review.

### 7.3 What leaves the old design and when

| today | gone after |
|---|---|
| pickle of `(header, args)` per message | 4a |
| `charm.arrays[aid][index]` lookup per message | 4a |
| greenlet per coroutine invocation | 4a |
| `CreateCallbackMsgExt` round trip per reduction result | 4a (Python side), 5 (C++ side) |
| `sections.py` and its trees | 4b |
| `CkReductionTypesExt` triplicated table | 4a (Python), 5 (C++) |
| `.copy()` of every received numpy array | 4a |
| migration refusing elements in sections | 4b |
| checkpoint that cannot work | 4c |
| Cython | 4d |
| `CMK_CHARM4PY` blocks in ck-core | 5 |

### 7.4 Cost targets and how they are measured

Protocol, on this Mac (8 cores, Apple Silicon), production reconverse build
(`--with-production`, no error checking, no tracing), `uptime` 1-minute load below 2,
nothing else running, Python 3.13 or 3.14 with the same interpreter for before and
after:

- `python tests/benchmark/pingpong.py +pe 1` (one process, both elements on PE 0):
  the printed "time per iter" is a round trip of two messages; per message = half.
  Two trials per run, five runs, report the median of the ten trials. Same with `-t`
  (coroutine) and with `-n 1048576` (1 MB numpy payload, which measures the receive
  copy).
- `lcrun -n 2 python tests/benchmark/pingpong.py +pe 2` for the cross-process number,
  which is dominated by LCI over TCP loopback on this Mac (about 45 us one way,
  machines/mac.md) and is reported for completeness, not as a target.
- `tests/charm++/ckdyn/pingpong +pe 1` with the C++ binding: the lower bound that no Python
  binding can beat; the gap between it and the Python number is the binding's cost.
- `python tests/reductions/bench_reductions.py +pe 1` as printed (5000 sums of 20000
  doubles), reported as time per reduction.

Step 0 ran on 2026-11-07 (BENCHMARKS.md). Measured today, per message, one PE: plain
2.27 us, coroutine 6.14 us, plain 1 MB 59.6 us, 100 KB 9.97 us; two processes 26 us
plain (LCI over TCP loopback); reductions 120 us per 20000-double sum. The components,
measured on the same build: C++ 1D-array ping-pong 0.18 us per message (the floor);
one 1 MB `memcpy` 20.4 us; a pickle round trip of one int 0.66 us; a greenlet created
per call 2.7 us against 0.37 us for a switch in and out of a pooled greenlet; a bound
method call 0.03 us; today's `__getattr__` proxy call 0.20 us.

Targets (section 0): plain 2.27 → 1.0 us, coroutine 6.14 → 1.4 us, 1 MB with no
receive copy (the printed time for `-n 1048576` must be within 10% of the `-n 0` time
plus one 1 MB `memcpy`, so about 23 us per message). Where the targets come from: the
floor is 0.18 us; the thin path replaces the pickle of the argument (0.66 us) with one
slot write and one slot read; the binding adds a GIL acquire, one vectorcall through
`ProxyMethod`, one message allocation and one hook call, together estimated at 0.5 us,
which leaves 0.3 us of margin in the 1.0 us target. The coroutine path in
`pingpong.py -t` never blocks (tests/benchmark/pingpong.py:38-46), so it adds only the
pooled switch pair of 4.5, measured at 0.37 us; today it pays 2.7 us for a greenlet per
call plus the rest of the coroutine bookkeeping. The 0.5 us binding estimate is the
one unmeasured component; step 4a measures it. If step 4a measures above 1.0 us plain,
the profile (`perf` on Linux, `sample` on macOS, with `charm.options.profiling` off)
decides between the GIL acquire cost, nanobind overhead and the builder before
anything else changes.

Decision 2026-11-08 (review F8): the coroutine result of step 4a, 1.48 us per message
at one PE, is accepted as meeting the 1.4 us target, and the performance side of the
step 4a gate is closed. The remaining Python pause path is not moved into C++; that is
revisited only after step 7, if the GIL work changes the path anyway.

---

## 8. Risks and open questions

### 8.1 Ownership of Layer 2 inside recharm

CkDyn is about 2,500 lines of C++ in `src/ck-core` whose only in-tree user is a test.
The risk is the one the Ext API realized: nobody on the Charm++ side runs charm4py, so
a change in `ckarray.C` or `ckcallback.C` breaks the binding silently. What the design
does about it: CkDyn compiles on every target (3.1); `tests/charm++/ckdyn` runs in
`reconverse-ci.yaml` on every pull request (3.16); the hooks and message format are
behind a versioned C ABI (`CKDYN_API_VERSION`) so the binding can detect a mismatch
at import; and `doc/internals/ckdyn.md` is the chapter a Charm++ maintainer updates in
the same pull request (charm/AGENTS.md). What it cannot do: make someone own it. The
recommendation is that the Charm++ repository owns CkDyn (its tests, its chapter, its
API version) and the charm4py repository owns the binding, with the charm4py
reconverse CI job (already building `recharm` with the submodule) as the cross-check
that runs nightly rather than per pull request. Open question for Kale: who is the
named maintainer of CkDyn, and does the nightly cross-check belong in the charm or the
charm4py repository.

### 8.2 Other users of the Ext API, and classic

The survey (surveys/survey-ext-api.md section 4, 15 GitHub code-search queries)
found no consumer of the Ext API outside charm4py; the three non-fork repositories
that match contain copied Charm++ source trees, and no Julia or other binding exists.
Deleting the Ext API from `recharm` in step 5 therefore breaks nobody except a
charm4py older than step 4 built against a newer libcharm, which fails at import with
an undefined-symbol error (the `libcharm_version` check, charm.py:1038-1049, only
refuses a libcharm that is too old; the thin charm4py's `CkDynApiVersion` check is the
one that gives a clear message in the other direction). The frozen `classic`
branch keeps the Ext API, and the old charm4py keeps building against it; the
decision not to port CkDyn to classic is reversible if someone needs the thin design
on a classic-only machine, at the cost of porting the `init.C`, `ck.C` and
`ckarray.C` changes to a branch that takes fixes only. Open question: is there any
platform where the thin charm4py is wanted and reconverse is not available (Windows,
which `charm.py` still special-cases at lines 85-90 and 536-537, is the candidate).

### 8.3 The interpreter and several PEs per process

Today one PE per process is the only configuration exercised (2.3). The binding is
written to be correct with several PE threads (GIL taken per hook, per-thread
greenlet state), but until step 7 that configuration is untested, and with the GIL it
gains nothing. The free-threaded interpreter removes the limit but is young: nanobind
supports it, numpy 2.1+ ships free-threaded wheels, and `pickle` is thread-safe;
greenlet's status on a free-threaded interpreter has to be checked against its
changelog before step 7, and the other unknown is charm4py's own Python layer (dict mutation in
`charm.py` from several PE threads). A separate risk: `Py_BEGIN_ALLOW_THREADS` around
the scheduler means `KeyboardInterrupt` no longer interrupts Python; `SIGINT` goes to
Charm++'s handler, which is the behavior of every Charm++ program but a change for
charm4py users who press Ctrl-C in an interactive session.

Decision 2026-11-08 (review F14): the binding is not correct for several PEs per
process as written; holding the GIL while calling into the runtime allows a lock-order
inversion (4.4). Until step 7, one PE per process, enforced by `_ckdyn.start`. Step 7
must consider three options:

- (a) Release the GIL around runtime calls, after an audit of the runtime locks under
  which a hook can run (`CkNodeReductionMgr`'s `lockEverything`, the node queue, the
  location manager in SMP mode, CkMulticast). About 0.1 us per send.
- (b) "Drone mode": the extra PEs of a process hold no chares and only run tasks
  handed out by the single chare-hosting PE, so no hook runs on them while another PE
  of the process is inside Python.
- (c) The free-threaded Python build (PEP 703; supported in 3.14; a separate
  interpreter binary; a 5-10% single-thread cost; greenlet support to be checked).

Also for step 7: for the one-PE case, measure holding the GIL while the scheduler is
busy and releasing it only in the idle begin and end callbacks
(`CcdPROCESSOR_BEGIN_IDLE` / `END_IDLE`), which removes the two mutex operations per
message of the per-hook `PyGILState_Ensure/Release` and still lets other Python
threads run while the PE is idle.

### 8.4 Runtime behaviors the design depends on

- **QD on reconverse** can hang when the system is idle (charm_best_practices.md,
  2026-07-22). `tests/qd/qd.py` exercises exactly that path; if it fails at step 4a the
  cause is the runtime, not the binding, and the test survey shows that `qd.py`
  passes today in the reconverse CI job, so the hang is environment-dependent.
- **CkMulticast ordering on reconverse** was fixed in charm #4131; the C++ section
  test runs under `+randomized_msgq` to keep it fixed.
- **Zero-copy broadcasts** have no reconverse implementation
  (`CmiForwardNodeBcastMsg` and `CmiForwardProcBcastMsg` are stubs,
  contrib/reconverse/src/conv-rdma.cpp:27-29), so an `NCPY` slot in a broadcast is
  delivered as a copy, which is what the design specifies (3.6).
- **The `bindTo` plus fresh map race** (charm_best_practices.md 2026-07-24) is why
  `CkDynArrayOpts` has `group_dep`; the Python `Array(C, map=m)` call sets it to the
  map's gid by default.
- **In-memory checkpoint** (`CkStartMemCheckpoint`) is not built in any current
  configuration (no header defines `CMK_MEM_CHECKPOINT`); the design covers it in
  principle and lists it as untested.
- **Shrink/expand on reconverse** does not exist: the ck-core and ck-ldb half is in
  the tree under `CMK_SHRINK_EXPAND` (ckcheckpoint.C, CentralLB.C, manager.C, init.C),
  but the symbols it expects from Converse (`Cmi_isOldProcess`, `Cmi_myoldpe`,
  `ConverseCleanup`) were defined in the classic netlrts machine layer that #4056
  removed, and nothing in `contrib/reconverse` defines them, so `--enable-shrinkexpand`
  cannot link on `recharm`. Step 8 is the reconverse work; the binding is ready for it
  as soon as pup and `AtSync` exist (step 4c).
- **Device buffers on reconverse** exist in the tree (ckrdmadevice.C:61-994) but could
  not be exercised on this Mac; step 3's GPU tests on Delta are the first run.
- **Registration limits**: 65536 entries per process and 256 message types (3.5). A
  program with many thousands of Python chare classes times methods would hit the
  first; nothing in the current user base is close.

### 8.5 Packaging

Reconverse wheels do not exist today; LCI's fabric backend ties the wheel to a
libfabric or UCX build. The plan is a TCP/libfabric-bundled wheel for laptops and
small clusters and source builds for production machines, which matches how the
classic wheels are used today (surveys/survey-build.md section 8: wheels are
`netlrts`, not `ucx` or `mpi`). Open question: whether to publish both a
`charm4py` (reconverse) and a `charm4py-classic` package during the transition.

### 8.6 Smaller open questions

1. Decided 2026-11-07 (Kale): `Chare(C)` and `Chare(C, onPE=p)` become real
   `ChareDyn` chares (5.4 item 13). The representation of a *migratable* singleton is
   postponed: a one-element array is the only form Charm++ offers today, and a better
   one is a Charm++ design question, not a binding question. Until then
   `Chare(C, migratable=True)` keeps the one-element-array form. No test or library
   code needs it: `tests/migration` creates arrays only, and the two singleton
   creators in the library (`PoolScheduler`, charm.py:644; Ray actors, ray/api.py:32)
   never migrate.
   Amended 2026-11-07 22:15 PST (review F6): a chare created with `CkCreateChare`
   has two identities, the virtual id (`-(creatorPe+1)`, VidBlock index) that the
   creator's proxy holds for the chare's lifetime and the real id (`pe`, `objPtr`)
   that `thishandle` inside the chare holds. Proxies built from the two compare
   unequal (channels never establish, `when` conditions and user dictionaries keyed
   by proxy break), and every send through the virtual id from a third PE is
   forwarded by the creator, two hops for life. Step 4a therefore keeps the
   one-element-array form for every `Chare(...)`. The real-chare singleton needs a
   Charm++-side mechanism that lets the creator know the final id at creation
   (for example an id assigned by the creator and installed on the destination PE
   by the creation message, the same mechanism a migratable singleton needs), and
   `thisIndex`/`proxy[0]` compatibility decided explicitly.
   Decision 2026-11-08 (review F11): singletons remain one-element arrays, so a restart
   with a different PE count loses none. Which PE hosts a real singleton chare whose
   PE disappeared is postponed. Note for later: if real singletons come, the host PE
   must be chosen by a deterministic rule (for example old PE modulo new PE count),
   because every PE resolves proxies without communication; and a helper singleton
   that should move with an array element is better expressed as a future or callback
   owned by the element than as a co-located chare.
2. Should the message header carry the ep's `user_id` (4 bytes per message) or should
   the binding index its table by Charm++ ep only? Recommendation: carry it; it costs
   nothing and lets the binding's table be sparse for late-registered classes.
3. Is `1 MB` the right default zero-copy threshold on a fabric where registration
   costs more than a copy below some size? Measure on Anvil in step 3; the option is
   there so the default can change per machine.
4. The `_bcast_exc_reducer` and `gather` foreign reducers run Python at every tree
   level; a C++ gather (concatenate contributions in index order) would avoid that
   for `ret=True` broadcasts. Not in scope; noted for step 7.
5. Whether to keep `appspector`'s per-entry Python timers or move them to
   `CkDynTraceBegin/EndEntry` once step 7 adds it. Decision 2026-11-08 (review F12):
   the coroutine brackets are now step 4a follow-up work (3.13), not step 7.

### 8.7 Stretch goal: bindings for languages other than Python

CkDyn is language-neutral by construction: its ABI is C (3.17), every binding
contact is a function pointer in `CkDynHooks` (3.3), objects are reached through an
opaque handle, and the C++ test binding (3.16) is a second client that keeps the
Python-specific parts out. Kale asked (2026-11-07) that other languages be in scope as
stretch goals. What each would need beyond CkDyn, with the Python answer for
comparison:

| language | object model and dispatch | threads and blocking | pup | notes |
|---|---|---|---|---|
| Python (this design) | handle = `PyObject*`, nanobind vectorcall | greenlet pool, GIL per hook | pickle | the first client |
| Julia | handle = a `Ref` kept alive in a root set; `@cfunction` hooks; dispatch through a `Vector{Function}` by `user_id` | Julia `Task`s are not migratable between processes either; blocking on a future suspends the task, same rules as 4.5 | `Serialization.serialize`; arrays are zero-copy through the BUFFER slot | no GIL; Julia threads can map to PE threads directly, which is the configuration 8.3 cannot have for Python until the free-threaded interpreter |
| Rust | handle = `Box<dyn DynChare>`; the hook table is a `static` of `extern "C" fn`s; a `#[derive(Entry)]` macro builds `CkDynClassDesc` tables at compile time | no coroutines needed for plain entries; `async` entries need an executor per PE | `serde` for migration and checkpoint | the closest to C++ in cost; a Rust binding is mostly a safe wrapper over 3.17 plus a macro |
| C | the `CkDynClassDesc` tables are written by hand; handle = `void*`; the C++ test binding of 3.16 is already this binding in C++ clothing | none. Decision 2026-11-08 (review F15): CkDyn only passes the `CKDYN_EP_THREADED` flag through and never creates a `CthThread`, so a C binding cannot block in an entry method until the deferred `CKDYN_EP_CTHREAD` option exists | a user pack/unpack pair | would let a C program use Charm++ without charmxi; the smallest stretch goal |
| Lua / other embeddable interpreters | handle = a registry reference; hooks call `lua_pcall` | Lua coroutines are first class and cheap; no GIL | the interpreter's serializer | same shape as Python with cheaper coroutines |
| C# (.NET) | handle = a `GCHandle` (normal, so the collector keeps the object and may move it; the pointer is the handle's integer, not the object's address); hooks are `[UnmanagedCallersOnly]` static methods whose addresses go into `CkDynHooks`; the runtime calls them on PE threads, which .NET attaches on first entry; dispatch through a per-class delegate array by `user_id` | no GIL: PE threads run managed code concurrently; blocking entries map to `async` methods whose continuations the binding resumes from `hooks.future` (a suspended `async` method is a heap state machine that cannot move between processes, so the migration rule of 4.5 applies unchanged) | `MemoryPack`, `MessagePack` or `System.Text.Json` for pup; arrays and `Span<T>` are pinned with `fixed` or `GCHandle.Alloc(Pinned)` for the duration of a BUFFER copy, and a received BUFFER view is a `ReadOnlySpan<T>` over message memory while the view lives | NativeAOT can produce a single executable that links libcharm statically; `libcharm` must then be loaded with `NativeLibrary.Load` or linked at build time; the `CkDynStart` path (ConverseInit never returns) runs on the main thread, which is fine for a console program |

Julia's own parallelism (added 2026-11-07, Kale's question). The table row covers
sequential Julia inside entry methods. Julia has three parallel layers, and each
meets CkDyn differently. (a) Threads (`Threads.@spawn`, `@threads`): Julia tasks run
on Julia's thread pool with its own work-stealing scheduler, and the pool coexists
with the PE threads in the same process the way numpy's threads coexist with
charm4py's PE today; a hook called on a PE thread enters Julia through
`@cfunction`, which since Julia 1.9 adopts a foreign thread, and the binding must
mark the PE thread GC-safe (`jl_gc_safe_enter`) whenever it returns to the Charm++
scheduler, or Julia's stop-the-world collector waits forever for a PE that is idle
in C. A task is not migratable, so 4.5 applies. (b) Distributed.jl (`addprocs`,
`RemoteChannel`, `pmap`, Dagger.jl on top): a process model with its own TCP
transport and launcher. Interoperation is possible at the transport: a custom
`ClusterManager` whose `launch` is the Charm++ job and whose `connect` returns IO
streams carried by CkDyn group messages, so a Distributed program runs on
Charm++'s launcher and network with no change to the program; it gains nothing
from Charm++'s object model, since Distributed is RPC plus futures, the same shape
as charm4py's pool and futures. (c) MPI.jl: an MPI Julia program can call a
Charm++ library in phases through the existing MPI interoperation
(`CharmLibInit`, src/ck-core/mpi-interoperate.C, reconverse #288), which is how a
CkDyn-based Julia library would be used from an MPI program. The combination that
uses Charm++ for what it is good at is (a) with CkDyn objects as the unit of
migration and load balancing and Julia tasks as the sequential bodies, which is
the Python design with a GIL-free interpreter.

Requirements on CkDyn that these add, all small and all consistent with the current
design: (a) nothing in `ckdyn.h` may name a Python or C++ type (3.17 already holds
this); (b) the hook table is per process, so two bindings in one process (Python and
Julia together) would need a hook table per `CkDynClassDesc`, which is a one-field
change (`binding_data` already exists; the hooks pointer would join it) and is worth
making in step 1 so that it never has to be retrofitted; (c) error reporting
(`hooks.error`) stays a string plus a code; (d) the test binding of 3.16 is the
conformance test any new binding runs first. None of this is scheduled; the C binding
is the one that would cost the least and prove the ABI, and could be a student
project once step 2 lands.

Decision 2026-11-08 (review F15): the ABI work is split.

Done before step 2 freezes `CKDYN_API_VERSION` 1:

- a `size` field in `CkDynHooks`, so that later hooks are compatible additions;
- fixed-width integer types for the kind and enum fields of structs copied into
  messages (`CkDynTarget`, `CkDynCallback`);
- documentation that binding handles must be non-NULL, since NULL means "not
  constructed" for a branch outside a constrained group (3.6);
- the `array.array` typecode lives in tag bits, not in the reserved slot field
  (already so in step 4a);
- documentation that views of REDRESULT and SYS messages are valid only during the
  hook, because their slots are synthesized on CkDyn's stack.

Deferred until a second binding exists (compatible additions once the size field
exists):

- a binding id namespace for futures, reducers and readonlies, so two bindings in one
  process do not receive each other's deposits and foreign reductions;
- a `CKDYN_EP_CTHREAD` option that runs an entry in a Converse thread, as charmxi
  does for `[threaded]`;
- a start-up option to skip Charm++'s signal handlers (Julia, the JVM and .NET use
  SIGSEGV or other signals themselves).

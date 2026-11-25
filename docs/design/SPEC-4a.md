# Step 4a specification: the `_ckdyn` binding and the Python cut-over

Written 2026-11-07 by the design session. This is the contract between the two
halves of step 4a (DESIGN.md section 7.1, step 4a): the nanobind extension module
`charm4py._ckdyn` (section 2) and the Python package that uses it (section 3). It
refines DESIGN.md sections 4 and 5; where it differs from DESIGN.md, this file wins
for step 4a and DESIGN.md is updated afterwards.

Repository and branch: `~/software/charm4py-thin/charm4py`, branch `thin`, created
from `main` at c5b288f. The Charm++ side is `~/software/charm4py-thin/charm`
(branch `ckdyn`, PR charmplusplus/charm#4193); its C API is
`charm/src/ck-core/ckdyn.h` and its internals chapter `charm/doc/internals/ckdyn.md`.
Read both before writing code. The libcharm to link is the `charm4py`-target build
`charm/reconverse-darwin-arm8-charm4py` (built with `REFNUM_TYPE="unsigned int"`;
its `include/ckdyn.h` is a copy of the source header, refreshed by rebuilding the
`ck` target).

Conventions kept from today's charm4py, because tests and library code rely on
them: `charm.groups[gid]`, `charm.arrays[aid][index]`, `charm.chares[cid]`,
`charm.entryMethods[ep]`, `charm.classEntryMethods`, `charm.proxyClasses`,
`obj._numthreads`, `obj.thisProxy`, `obj.thisIndex` (an int for groups, a tuple
for arrays), proxy attributes `gid`, `aid`, `cid`, `elemIdx`, `ndims`, `issec`,
`section`, bound proxy method attributes `ep` and `__self__`, the `+libcharm_interface`
option accepted and ignored, `charm4py.threads.Future`, `charm4py.wait.parse_cond_str`.

## 1. Message convention (shared by both halves)

A CkDyn message carries typed slots (ckdyn.h `CkDynSlot`). The binding lays the
slots of an entry-method message out in this order:

1. If the header has `CKDYN_MSG_BLOCK`: slot 0 is the FUTURE slot of the future that
   receives the return value (or the creation acknowledgement, with
   `CKDYN_MSG_CREATION`). It is first because CkDyn's `findFuture` takes the first
   FUTURE slot for creation acknowledgements, and a constructor argument may itself
   be a future.
2. One slot per positional argument, in order (table below).
3. Zero or more BUFFER slots with `bflags & CKDYN_BUF_OOB`: the out-of-band buffers
   that `pickle.dumps(..., protocol=5, buffer_callback=...)` produced while pickling
   the arguments of row "anything else", in the order the callback received them.
4. If any argument was pickled: one TAIL slot (added by `CkDynMsgReserveTail`); the
   tail holds `pickle.dumps(tuple_of_pickled_args, protocol=5, buffer_callback=cb)`.

The receiver classifies slots by type and flags, never by assuming counts: slot 0
with BLOCK is the block future; a TAIL slot marks the tail; BUFFER slots with OOB
are the pickle buffers (passed to `pickle.loads(tail, buffers=...)` in slot order);
every other slot is a positional argument. A NONE slot with `tag == 1` is a
positional argument whose value is the next item of the unpickled tail tuple
(`tag == 0` is Python `None`).

Argument conversion, sender side, by exact type (`Py_TYPE(o) == ...`; subclasses go
to the last row):

| Python value | slot | details |
|---|---|---|
| `None` | NONE, tag 0 | |
| `bool` | BOOL | tested before `int` |
| `int` | I64 (or U64 when it fits only unsigned); overflow: last row | `PyLong_AsLongLongAndOverflow` |
| `float` | F64 | |
| `bytes` | BYTES, tag 0 | `CkDynMsgAddBytes` |
| `str` | BYTES, tag 1 | UTF-8 |
| `numpy.ndarray`, dtype without objects, C- or Fortran-contiguous | BUFFER | `elem` from the dtype (table below), `ndim`, `shape`; `CKDYN_BUF_FORTRAN` for Fortran order; a non-contiguous array is made contiguous first (`PyArray_GETCONTIGUOUS`); dtypes outside the table (structured, datetime, strings) go to the last row |
| `array.array` | BUFFER, tag 2, `ndim = 1` | `elem` from the typecode; the typecode letter is stored in the slot's `reserved` field (uint16) so it round-trips exactly |
| `charm4py.threads.Future` with `store == False` | FUTURE | `(future.src, future.fid)` |
| a proxy (`_ckdyn.Proxy` or a subclass) that is not a section proxy | PROXY | the proxy's target; `tag` bit 0 set when the proxy has no element (a broadcast proxy) |
| anything else (including store futures, section proxies, exceptions, lists, dicts, objects) | NONE, tag 1, plus the tail | pickled together, protocol 5, out-of-band buffers as OOB BUFFER slots |

Element kinds (`CKDYN_ELEM_*`) and numpy dtypes: I8 int8, I16 int16, I32 int32,
I64 int64, U8 uint8, U16 uint16, U32 uint32, U64 uint64, F16 float16, F32 float32,
F64 float64, C64 complex64, C128 complex128, BOOL bool, BYTE: raw bytes. `array.array`
typecodes map by size and signedness (`b` I8, `B` U8, `h` I16, `H` U16, `i` I32,
`I` U32, `l`/`q` I64, `L`/`Q` U64, `f` F32, `d` F64, `u`/`w` go to the last row).

Receiver side: NONE → `None` or the next tail item; BOOL → `bool`; I64/U64 → `int`;
F64 → `float`; BYTES tag 0 → `bytes`; BYTES tag 1 → `str`; BUFFER tag 2 →
`array.array(typecode)` built with `frombytes` (a copy); BUFFER otherwise →
`numpy.ndarray` view over the message bytes, dtype from `elem`, shape from the slot,
order from `CKDYN_BUF_FORTRAN`, base object a `_ckdyn.MsgRef` that holds one
`CkDynMsgRetain` and releases it when collected (one MsgRef per delivered message,
shared by all its views); the view is writable for a point-to-point delivery and
read-only for a broadcast (`CKDYN_MSG_BCAST` set, or delivered through
`deliver_many`); if `charm.options.copy_received_arrays` is true the view is copied
instead and no MsgRef is created. FUTURE → `threads.remote_future(pe, fid)`
(section 3.4). PROXY → a proxy object of the right Python class (section 2.5).

Reduction results: a view of kind `CKDYN_MSGKIND_REDRESULT` has one BUFFER slot
(data of a built-in reducer, `elem` from the reducer table, `ndim` 0) and possibly
a FUTURE slot. Its value is: `None` if `len == 0` (a nop reduction), a Python
scalar (`int`, `float`, `bool`) if the data holds exactly one element, else a numpy
array of the element kind (today's behavior, charmlib_cython.pyx:1081-1096). A
result of the foreign reducer arrives as a DYN view whose header has the
binding-private flag `PYMSG_REDRESULT` (section 1.1) and whose `user_id` is the
binding's reducer id; its value is the first positional argument of that payload,
converted by the table, and then `postprocess` of that reducer applied when the
reducer has one.

### 1.1 Binding-private header flags

ckdyn.h reserves bits 12-15 of `CkDynHdr.flags` for the binding
(`CKDYN_MSG_BINDING_MASK`, added 2026-11-07). The Python binding defines:

- `PYMSG_REDRESULT = 1 << 12`: the payload is the result of a foreign (Python)
  reduction; `hdr->user_id` is the reducer id. Set by the reduce hook on the result
  builder.
- `PYMSG_SINGLE = 1 << 13`: a constructor message of the legacy one-element-array
  singleton (`Chare(C, migratable=True)` or internal use); the constructed element
  gets `thisProxy` bound to its own index (today's `single` header key).

## 2. The extension module `charm4py._ckdyn`

Source: `charm4py/src/ckdyn_py.cpp` (the binding) and `charm4py/src/objstore_py.cpp`
(the port of `c_object_store.pyx`, same module, section 2.9). Built with nanobind
(3.1 is installed in the venv) and `greenlet/greenlet.h` (the greenlet C API,
`PyGreenlet_Import()` at module init). Links `libcharm` from the build directory.
Includes `ckdyn.h` only (no Charm++ header). Everything below is the whole surface
the Python package uses.

### 2.1 Module state

```c++
struct MethodEntry {           // one per Charm++ entry index
  PyObject *func;              // unbound Python function (strong ref), NULL for entries with no body
  PyObject *em;                // the charm4py EntryMethod object (strong ref)
  PyObject *name;              // "Class.method"
  uint32_t  flags;             // CKDYN_EP_* as registered
  bool      has_when;          // em.when_cond is not None
  int       argcount;          // positional parameters after self
  PyObject *argnames;          // tuple of str
  PyObject *defaults;          // tuple
};
std::vector<MethodEntry> methods;   // indexed by Charm++ ep; user_id == ep
struct ClassEntry { PyObject *cls; CkDynKind kind; int first_ep; int nentries; PyObject *proxy_class; };
std::vector<ClassEntry> classes;    // indexed by CkDyn chare index
```

Module globals set by `init`: the `charm` object, the `options` object, and the
Python callables the hooks call (looked up once by name on `charm`, held as strong
refs): `registerMainModule`, `_construct`, `_mainchare`, `_invoke`,
`_process_exception`, `_deposit_future`, `_reduce_foreign`, `_pack`, `_unpack`,
`_pack_readonlies`, `recvReadOnly`, `_map_proc_num`, `_resume_from_sync`,
`_destruct`, `triggerCallable`, `callHandler`, `_error`; also `threads.Future`
(the exact type for the FUTURE row) and `threads.remote_future`. `g_profiling` is
read from `options.profiling` in `start()` and never changes afterwards.

Per PE (one PE per process in step 4a; still keep it in a struct so step 7 can make
it `thread_local`): the idle worker greenlets (section 2.6), the main greenlet
(`PyGreenlet_GetCurrent()` captured in `start()`), `last_ep` (the entry the fast path
last ran, for error messages).

### 2.2 Hooks (C++ side of the `CkDynHooks` table)

Every hook takes the GIL (`nb::gil_scoped_acquire`), does its work, and never lets a
Python exception escape: an exception from a Python callable other than the entry
method itself is reported with `charm._error(str)` and `CkDynAbort`. The entry
method's own exceptions go to `charm._process_exception` (fast path) or are handled
inside `_invoke` (slow path).

- `register_classes(global)`: calls `charm.registerMainModule()`.
- `mainchare(bd, handle_out, target, argc, argv)`: `obj = charm._mainchare(Target(t), [argv...])`;
  `*handle_out = obj` with a new strong reference.
- `construct(bd, target, view)`: builds `header, args` from the view exactly as the
  slow delivery path does (section 2.3), `obj = charm._construct(Target(t), ep, header, args)`
  where `ep = view->hdr->ep`; returns a new strong reference. The CONSTRAINED flag's
  PE-list slot (the last BYTES slot) is not an argument: strip it when the header
  has `CKDYN_MSG_CONSTRAINED`.
- `destruct(bd, handle)`: `charm._destruct(obj)` then `Py_DECREF(obj)`.
- `deliver(bd, handle, ep, user_id, ep_flags, view)`: section 2.3.
- `deliver_many(bd, handles, n, ep, user_id, ep_flags, view)`: converts the view once
  (read-only buffers), then for each handle: `CkDynCallstackPush(target of obj)`,
  deliver as in 2.3 reusing the converted args (the same Python objects are passed
  to every element, as today's broadcast path does), `CkDynCallstackPop`. The target
  of an object is its `_ckdyn_target` attribute (a `Target`); read it once per element.
- `pack_size(bd, handle, reason)`: on the sizing pass (`reason & CKDYN_PUP_SIZING`)
  calls `data = charm._pack(obj, reason & 0xff)`; `None` means refused →
  `CKDYN_PACK_REFUSED`; otherwise keeps the bytes object in a per-object side table
  keyed by handle and returns its length. On the packing pass returns the kept
  length (calling `_pack` again if nothing is kept).
- `pack(bd, handle, reason, buf, len)`: copies the kept bytes, drops them from the side table.
- `unpack(bd, target, reason, buf, len)`: `obj = charm._unpack(Target(t), reason, memoryview_of(buf, len))`,
  returns a new strong reference.
- `pack_readonly_size/pack_readonly`: `charm._pack_readonlies()` returns bytes (kept
  between the two calls like `pack_size`/`pack`); `unpack_readonly`: `charm.recvReadOnly(bytes)`.
- `reduce(global, reducer_id, contribs, n, result)`: for each contribution view
  takes its first positional argument (converted by the table; buffers are copied,
  since the contribution messages do not outlive the hook), builds a list, calls
  `value = charm._reduce_foreign(reducer_id, values)`, and writes `value` into the
  result builder as one positional argument (table row by type; a pickled value
  gets a NONE tag 1 slot plus tail) with `CkDynMsgSetFlags(result, PYMSG_REDRESULT)`.
- `future(global, pe, id, view)`: `value` = the view's value: a REDRESULT view as in
  section 1; a SYS view → `None`; a DYN view with `PYMSG_REDRESULT` → first positional
  argument and `rid = hdr->user_id`; any other DYN view → its first positional
  argument, or `None` if it has none. Calls `charm._deposit_future(id, value, rid)`
  with `rid = -1` when not a foreign result. Buffers in the value are copied (the
  message is freed by CkDyn after the hook).
- `map_proc_num(bd, handle, aid, ndims, index)`: `charm._map_proc_num(obj, aid, tuple(index))`.
- `resume_from_sync(bd, handle)`: `charm._resume_from_sync(obj)`.
- `error(global, what)`: `charm._error(what)`.
- `section_joined`: NULL (step 4b).

### 2.3 Delivery: fast and slow paths

```
deliver(bd, handle, ep, user_id, ep_flags, view):
  MethodEntry &m = methods[ep]; obj = handle
  hflags = view->hdr ? view->hdr->flags : 0
  if (!g_profiling && !m.has_when && !(ep_flags & CKDYN_EP_THREADED) && hflags == 0 && m.func):
      // fast path
      convert positional arguments into a stack array args[1..n], args[0] = obj
      last_ep = ep
      r = PyObject_Vectorcall(m.func, args, n + 1, NULL)
      if (!r) charm._process_exception(obj, ep, exc)   // fetch the exception first
      else Py_DECREF(r)
      if (obj has attribute _cond_next and it is not None) obj.__flush_wait_queues__()
  else:
      header = {}; if BLOCK: header[b'block'] = remote future from slot 0
      if BCAST: header[b'bcast'] = True; if BCASTRET: header[b'bcastret'] = True
      if CREATION: header[b'creation'] = True; if RAY: header['is_ray'] = True
      if PYMSG_REDRESULT: header[b'custom_reducer'] = hdr->user_id
      if PYMSG_SINGLE: header[b'single'] = True
      args = list of positional arguments (REDRESULT view: [value] or [] when empty)
      charm._invoke(obj, ep, header, args)
```

`_cond_next` is read with `PyObject_GetAttr`; every chare has it. The fast path
must be re-entrant (an inline send delivers inside another delivery) and must not
hold any C++ state across the vectorcall other than stack variables.

### 2.4 Sends: `ProxyMethod` and `BoundProxyMethod`

Two types written with the raw C API inside the module (nanobind does not expose
`Py_TPFLAGS_METHOD_DESCRIPTOR`):

- `ProxyMethod`: created by `make_proxy_method(ep, argcount, argnames, defaults, name, qualname)`.
  Has `Py_TPFLAGS_METHOD_DESCRIPTOR`, a vectorcall slot, and `tp_descr_get` that
  returns a `BoundProxyMethod(proxy, self)`. Attributes: `ep`, `__name__`,
  `__qualname__`. Stored in the generated proxy class dict under the method name,
  so `proxy.method(a, b)` runs the vectorcall with `proxy` as the first argument
  and no bound object is allocated.
- `BoundProxyMethod`: fields `proxy` (strong), `pm` (strong); vectorcall forwards to
  `pm` with `proxy` prepended; attributes `ep`, `__self__` (the proxy), `__name__`,
  `__qualname__`; `__eq__`/`__hash__` by `(proxy, ep)`. It is what callback targets
  are (`self.thisProxy.done`), and what `charm.contribute`, `start_qd` and
  `Callback` consume.

The vectorcall `(proxy, *args, **kwargs)`:

1. `kwargs`: pop `ret`, `awaitable`, `is_ray` (all default false). Map remaining
   kwargs onto missing positional parameters by `argnames`, then `defaults`
   (chare.py:321-336 today); an unknown keyword or a missing parameter raises
   `TypeError` with the parameter name.
2. `future = None`. If `ret` or `awaitable`: `future = charm.Future()` (call into
   Python; this is the slow case). If `is_ray`: `future = charm.createFuture(store=True)`,
   appended as the last positional argument, and the RAY flag set (as chare.py:757-761).
3. Flags: BLOCK when `future` was created for ret/awaitable; BCAST when the proxy
   has no element and its kind is GROUP, NODEGROUP, ARRAYMAP or ARRAY; BCASTRET when
   BCAST and `ret`.
4. Build: `b = CkDynMsgBegin(nslots_hint, data_hint, 0, 0, -1)`; set flags; if BLOCK
   add the FUTURE slot first; then the arguments per section 1; `m = CkDynMsgFinish(b, ep)`.
5. Send: `CkDynSendTarget(&t, ep, m, NULL)` for an element target; for a broadcast
   `CkDynSendGroup(gid, -1, ...)`, `CkDynSendNodeGroup(gid, -1, ...)`, or
   `CkDynSendArray(aid, 0, NULL, ...)`. A section proxy raises `NotImplementedError("sections: step 4b")`.
6. Return `future` (a `threads.Future` or `None`).

Also exported: `send(proxy, ep, args_tuple, flags, future)` as the slow-path entry
for internal uses, with the same steps 3-5.

### 2.5 Proxies and targets

`Target`: a nanobind class wrapping a `CkDynTarget` by value. Read-only properties:
`kind`, `chare_idx`, `gid`, `aid`, `pe` (group/nodegroup element PE or node, -1 for
none), `index` (tuple for arrays), `cid` (`(pe, ptr)` as ints for chares). Pickle
support (`__getstate__`/`__setstate__`), `__eq__`, `__hash__`, `__repr__`.

`Proxy`: the nanobind base class of every generated proxy class (`nb::dynamic_attr()`
so Python code can set `creation_future` on an instance). C fields: a `CkDynTarget`
and a bool `has_elem`. Constructors: `Proxy(kind: int, id: int, ndims: int)` where
`id` is the gid or aid (CHARE and MAINCHARE proxies are built by
`Proxy._from_target(target)` instead). Methods and properties, all cheap:

- `gid`, `aid`, `cid`, `ndims`: as today's attributes (`gid` for GROUP/NODEGROUP/ARRAYMAP,
  `aid` for ARRAY, `cid` for CHARE/MAINCHARE); accessing the wrong one raises `AttributeError`.
- `elemIdx`: groups: the PE or `-1`; arrays: the index tuple or `()`; chares: `None`.
- `issec`: always `False`; `section`: always `None` (step 4b replaces both).
- `_target` → a `Target` copy; `_set_elem(pe)`, `_set_index(tuple)`, `_clear_elem()`
  mutate the proxy (used right after constructing a copy, chare.py `group_proxy_elem`
  / `array_proxy_elem`).
- `__getstate__`/`__setstate__` (the target and `has_elem`); `__eq__`/`__hash__` by
  target and `has_elem` (so the Python classes do not need their own, but may keep them).

A PROXY slot is received as an instance of `classes[t.chare_idx].proxy_class`
(set by `set_proxy_class(chare_idx, cls)` right after the Python side generates the
class), created with `cls.__new__(cls)` and the C fields filled in; no `__init__` call.

`Callback`: a nanobind class wrapping `CkDynCallback`. Constructors:
`Callback.target(bound_proxy_method)` (TARGET, `bcast = not proxy.has_elem`),
`Callback.future(pe, fid)`, `Callback.ignore()`, `Callback.exit()`. Functions that
take a `target` argument (`contribute`, `start_qd`, `checkpoint`) accept a
`BoundProxyMethod`, a `threads.Future` (read `.src` and `.fid`), or a `Callback`.

### 2.6 Coroutines: the worker pool

`spawn(fn, *args) -> bool` and `resume(gr, value) -> bool` implement DESIGN.md 4.5
with the parent rules of today's threads.py:

- A worker is a greenlet created with `PyGreenlet_New(worker_main, current)`; its
  run function loops: run the task (`fn(*args)`), push itself on the idle list,
  switch to its parent passing the sentinel `DONE` (or `(EXC, exc)` when the task
  raised; the switcher re-raises it, as `gr.switch` propagates today); the value
  that comes back from the switch is the next task `(fn, args)`.
- `spawn`: pops an idle worker (or creates one), sets `worker.parent = current`
  (`PyGreenlet_SetParent`), switches to it with the task; returns `True` if the
  switch came back with `DONE`, `False` if the worker paused (any other value).
- `resume(gr, value)`: `r = gr.switch(value)`; returns `True` on `DONE`, else `False`.
  (`gr` is a worker greenlet; Python code obtained it from `getcurrent()` inside
  the task.) It does not touch `parent`.
- Pausing stays in Python (`threads.py pauseThread`: `main_gr.switch()` with the
  "parent is not main" re-parenting rule); `_ckdyn.main_greenlet()` returns the main
  greenlet captured in `start()`.
- A worker that is thrown `GreenletExit` (interpreter shutdown) leaves the loop.

Python uses the return values to maintain `obj._numthreads` (section 3.3).

### 2.7 Other module functions

```
init(charm, options)                       # records the objects and the callables of 2.1; installs CkDynHooks (CkDynInit)
start(argv: list[str])                     # CkDynAllowLateRegistration() if options.remote_exec; captures the main greenlet; g_profiling; releases the GIL; CkDynStart(argc, argv)
register_class(cls, kind, ndims, entries) -> (chare_idx, first_ep)
    # entries: list of (func_or_None, em, "Class.method", flags, has_when, argcount, argnames, defaults); entries[0] is the constructor
    # uses CkDynRegisterClass before start, CkDynRegisterClassLate after
set_proxy_class(chare_idx, cls)
api_version() / refnum_bytes() / has_device_support()
my_pe() / num_pes() / my_node() / num_nodes() / wall_timer() / exit(code) / abort(msg) / printf(text)
num_physical_nodes() / physical_node_id(pe) / physical_rank(pe) / num_pes_on_physical_node(node) / pes_on_physical_node(node) -> list / node_first(node)
topo_tree_edges(pe, root, pes_or_None, bfactor) -> (parent_or_None, children)     # CkDynTopoTreeEdges
make_proxy_method(ep, argcount, argnames, defaults, name, qualname) -> ProxyMethod
send(proxy, ep, args, flags, future)
create_chare(chare_idx, on_pe, args, flags, future) -> Target        # CkDynCreateChare; flags may include CREATION (future then required, BLOCK set too)
create_group(chare_idx, args, flags, future, pes_or_None) -> gid     # pes: constrained group, adds CONSTRAINED and the int32 BYTES slot last
create_nodegroup(chare_idx, args, flags, future) -> gid
create_array(chare_idx, dims_or_None, ndims, args, map_gid, use_at_sync, flags, future) -> aid
array_insert(aid, index, on_pe, args, flags, future, use_at_sync)    # never CREATION; a creation future travels as BLOCK
array_done_inserting(aid)
destroy_element(aid, index)
contribute(obj, data, reducer_type, target)
    # data: None for nop; a contiguous buffer object (numpy array, array.array, bytes) for a built-in type;
    # for the foreign type: (reducer_id, value) -> a payload with user_id = reducer_id and value as its one positional argument (section 1)
reducers() -> list[(name, type, elem)]      # CkDynReducers
foreign_reducer_type() -> int
reduction_number(obj) -> int
future_set(pe, fid, value)                  # one positional argument, CkDynFutureSet
start_qd(target)
at_sync(obj) / set_migratable(obj, flag) / migrate(obj, pe) / lb_instrument(flag) / callstack_push(obj) / callstack_pop(obj)
checkpoint(dirname, target)
spawn(fn, *args) -> bool / resume(gr, value) -> bool / main_greenlet()
last_ep() -> int
trace_register_user_event(desc, id) -> int / trace_begin_user_bracket(id) / trace_end_user_bracket(id)
ccs_register(name) / ccs_is_remote_request() / ccs_send_reply(bytes) / ccs_delay_reply() -> int / ccs_send_delayed_reply(token, bytes)
    # the C handler calls charm.callHandler(name, bytes)
call_after(tag, msecs)                      # CkDynCallAfter; the C function calls charm.triggerCallable(tag)
hapi_add_callback(stream, future)           # raises Charm4PyError unless has_device_support()
is_restarting() -> bool
```

`obj` arguments are chare instances; their target is the `_ckdyn_target` attribute
(a `Target`), set by the Python side in `_construct`/`_mainchare`/`_unpack`.

### 2.8 Errors

A nanobind exception translator maps a C++ `CkDynError` (thrown by the module's own
checks, for example a FUTURE slot in a message to a chare kind that cannot carry it)
to `charm4py.charm.Charm4PyError`. CkDyn's own aborts stay aborts.

### 2.9 The object store

`objstore_py.cpp` ports `c_object_store.pyx` and `.pxd` to nanobind classes
`MessageBuffer` and `CObjectStore` with the same method names, signatures and
semantics (including the numpy arrays it returns). `charm4py/object_store.py` and
`charm.py` import them as `from charm4py._ckdyn import MessageBuffer, CObjectStore`;
a one-line `charm4py/c_object_store.py` shim re-exports them so existing import
paths keep working.

## 3. The Python package

### 3.1 Files

| file | change |
|---|---|
| `charmlib/` (ccharm.pxd, charmlib_cython.pyx, generated .c/.html/.so), `c_object_store.pyx/.pxd`, `sections.py`, `setup.py` | deleted (`git rm`) |
| `c_object_store.py` | new shim (2.9) |
| `src/ckdyn_py.cpp`, `src/objstore_py.cpp`, `CMakeLists.txt`, `pyproject.toml` | new (section 4) |
| `charm.py` | rewritten in part (3.2) |
| `chare.py` | rewritten in part (3.5) |
| `entry_method.py`, `threads.py`, `reduction.py` | edited (3.3, 3.4, 3.6) |
| `__init__.py` | imports unchanged; add `NodeGroup` and `entry` exports (NodeGroup may be a stub raising NotImplementedError until step 7) |
| everything else (`wait.py`, `channel.py`, `pool.py`, `ray/`, `object_store.py`, `interactive.py`, `liveviz.py`, `appspector.py`, `loadfmt.py`, `charmrun/`) | unchanged, except imports of the object store |

### 3.2 `charm.py`

Delete: `recvReadOnly`'s pickling counterpart in `buildMainchare`, `recvChareMsg`,
`recvGroupMsg`, `arrayMapProcNum`, `recvArrayMsg`, `recvArrayBcast`, `unpackMsg`,
`packMsg`, `rebuild*`, `groupMsgBuf`, `arrayElemLeave`, the section half of
`contribute`, `combine`, `split` (both raise `NotImplementedError('sections: step 4b')`),
`startQD`'s four cases, `load_charm_library`, `lib_version_check`, `self.lib` and
every `self.lib.*` call, `CkContributeTo*`, `Ck*Send`, `self.ReducerType`.

Keep with the same names and behavior: `Options`, `Charm4PyError`, `register`,
`registerAs`, `register`, `start`, `registerInCharm`/`registerMainModule` (now
building entry lists for `_ckdyn.register_class`), `invokeEntryMethod`,
`handleGeneralError`, `prepareExceptionForSend`, `process_em_exc`, `recvReadOnly`,
`contribute` (non-section half), `waitQD`, `sleep`, `awaitCreation`,
`scheduleCallableAfter`/`triggerCallable`, `iwait`/`wait`, profiling and
`printStats` (drop the `self.lib.times` rows), topology (`getTopoTreeEdges`,
`getTopoSubtrees` reimplemented in Python over `_ckdyn.topo_tree_edges` as
charmlib_cython.pyx:652-689 did), `myPe`/`numPes`/host functions, `exit`, `abort`,
`addReducer`, `LBTurnInstrument*`, CCS functions, `CharmRemote` (minus
`_future_deposit_result`; `createArray`, `updateGlobals`, `rexec`, `eval`,
`propagateException`, `printStats`, `registerNewChareType(s)`, `notify_future_deletion`
stay), `get_future_value`/`getany_future_value`/`check_*_buffer` (Ray; unchanged).

New methods, the targets of section 2.2:

- `_mainchare(target, argv)`: today's `buildMainchare` without the readonly
  registration: creates the object (`object.__new__`, `Mainchare.initMember`,
  `Chare.__init__`, `obj._ckdyn_target = target`), `_createInternalChares()`, runs
  `__init__` through `em.run(obj, {}, [argv])`, stores `charm.chares[cid]`, returns `obj`.
- `_pack_readonlies()`: the dict of `readonlies` attributes pickled (today's
  buildMainchare lines 305-312), on PE 0; returns `bytes`.
- `_construct(target, ep, header, args)`: by `target.kind`: GROUP/NODEGROUP/ARRAYMAP →
  today's group branch of `recvGroupMsg` (`Group.initMember`, store in `charm.groups`);
  ARRAY → today's constructor branch of `recvArrayMsg` (`Array.initMember`, `single`
  from `header`, store in `charm.arrays`); CHARE → like Mainchare with the proxy of
  the CHARE kind, store in `charm.chares[cid]`. Sets `obj._ckdyn_target = target`,
  then `em.run(obj, header, args)`. Returns `obj`.
- `_invoke(obj, ep, header, args)`: today's `invokeEntryMethod` (when-condition
  enqueue, `em.run`, auto-flush), plus: if `b'custom_reducer' in header` apply the
  reducer's `postprocess` to `args[-1]` first (today's unpackMsg tail), and if
  `header.get('is_ray')` do today's `recvArrayMsg` dependency resolution before invoking.
- `_process_exception(obj, ep, exc)`: fast-path errors. `SystemExit` → `charm.exit(code)`
  with today's code normalization (entry_method.py:52-58); any other exception →
  today's `handleGeneralError` behavior using `exc.__traceback__` (print the
  traceback with the PE, then `abort`; in interactive mode forward it through
  `propagateException`). Factor `handleGeneralError` so both paths share the printing.
- `_deposit_future(fid, value, rid)`: if `rid >= 0`, apply that reducer's
  `postprocess` (if any) to `value`; then `self.threadMgr.depositFuture(fid, value)`.
- `_reduce_foreign(rid, values)`: `self.redMgr.reducers_by_id[rid](values)`.
- `_pack(obj, reason)`: today's `arrayElemLeave` minus the section abort, returning
  the pickled bytes, or `None` when `reason == CKDYN_PUP_MIGRATE and obj._numthreads > 0`
  (DESIGN.md 4.5 item 3; the C++ side turns `None` into `CKDYN_PACK_REFUSED`). Removes
  `obj` from `charm.arrays` only when the reason is MIGRATE. Deletes `_ckdyn_target`
  before pickling and restores it after (checkpoint keeps the object alive).
- `_unpack(target, reason, data)`: unpickle, `obj._ckdyn_target = target`, store in
  `charm.arrays[aid][index]` (or `charm.groups`/`charm.chares` by kind), call
  `migrated()` through the `migrated` EntryMethod (`em.run(obj, {}, ())`) for a
  MIGRATE reason; return `obj`.
- `_map_proc_num(obj, aid, index)`: `obj.procNum(index)`.
- `_resume_from_sync(obj)`: `self.invokeEntryMethod(obj, obj.thisProxy.resumeFromSync.ep, {}, [])`
  if the class defines `resumeFromSync`, else nothing.
- `_destruct(obj)`: remove it from `charm.groups`/`charm.arrays`/`charm.chares` if present.
- `_error(msg)`: print `"CkDyn error: " + msg` to stderr.

`contribute(data, reducer, target, chare, section=None)`: `section` not None or a
section proxy → `NotImplementedError('sections: step 4b')`. Else
`rtype, data = self.redMgr.prepare(data, reducer, chare)` and
`_ckdyn.contribute(chare, data, rtype, target)`.

`startQD(callback)` → `_ckdyn.start_qd(callback)`.

`registerMainModule()`: sets `_myPe`/`_numPes` from `_ckdyn`, the stdout/stderr
line buffering, the banner (interface name "ckdyn"), then `registerInCharm(C)` for
`register_order`. `registerInCharmAs(C, charm_type)`: builds the entry list from
`classEntryMethods[type_id][C]`: for each `EntryMethod em`: `func = getattr(C, em.name)`
(None for `migrated`/`AtSync` placeholders is fine: they are never sent), name
`C.__name__ + '.' + em.name`, flags `CKDYN_EP_THREADED` if the method has `_ck_coro`,
`has_when = em.when_cond is not None`, `argcount, argnames, defaults` from
`getEntryMethodInfo`. Kind: MAINCHARE → `CKDYN_MAINCHARE`; CHARE → `CKDYN_CHARE`;
GROUP → `CKDYN_ARRAYMAP` for `ArrayMap` subclasses else `CKDYN_GROUP`; ARRAY →
`CKDYN_ARRAY` with `ndims = 0` (any dimension; see the ckdyn.h change of
2026-11-07). Stores `C.idx[type_id]` (the chare index) and `em.epIdx`, fills
`charm.entryMethods`, generates the proxy class, calls `_ckdyn.set_proxy_class`.
Section proxy classes are no longer generated.

`start()`: unchanged up to `self.threadMgr.start()`; then `_ckdyn.init(self, self.options)`
(at `Charm.__init__` time is also fine) and `_ckdyn.start(sys.argv)`. The
`+libcharm_interface X` pair is removed from `sys.argv` and ignored.
`charm.options.local_msg_optim` and `local_msg_buf_size` stay as attributes, ignored
(a one-time warning if set true). `charm.options.pickle_protocol` default 5;
`charm.options.copy_received_arrays = False` and `zerocopy_threshold = 1 << 20` added.

`_registerInternalChares()`: registers `CharmRemote` (GROUP), `PoolScheduler`
(CHARE and ARRAY, since `Chare(PoolScheduler, onPE=0)` now creates a CHARE kind),
`Worker` (GROUP). No `SectionManager`. `_createInternalChares()`: `Group(CharmRemote)`,
the pool as today.

### 3.3 `entry_method.py`

`EntryMethod` keeps its fields. `_run` unchanged except that `charm.last_em_exec = self`
stays. `_run_th(obj, header, args, ret_fut=False)`:

```python
obj._numthreads += 1
if obj._numthreads == 1 and getattr(obj, 'migratable', False):
    _ckdyn.set_migratable(obj, False)
if _ckdyn.spawn(self._run, obj, header, args, ret_fut):
    obj._numthreads -= 1
    if obj._numthreads == 0 and getattr(obj, 'migratable', False):
        _ckdyn.set_migratable(obj, True)
```

`_run` must set `gr = getcurrent(); gr.obj = obj; gr.notify = self.thread_notify`
at its start (the attributes today's `_run_th` set on a fresh greenlet; pooled
workers are reused). `_run_th_prof` likewise with `gr.em_callstack = [self]`. The
`set_migratable` calls apply only to array elements (the only kind with
`migratable`); a chare whose `migratable` attribute is False keeps it False. Add
`entry(**attrs)` as a no-op decorator that records the attributes on the function
(step 7 uses them).

### 3.4 `threads.py`

`Future` unchanged in API. `send(result)`: store futures as today; else
`_ckdyn.future_set(self.src, self.fid, result)`. `getTargetProxyEntryMethod` removed
(callers now pass the Future itself). Add module function
`remote_future(pe, fid) -> Future`: `Future.__new__(Future)` with `fid`, `src = pe`,
`gr = None`, `nvals = 1`, `values = []`, `blocked = False`, `gotvalues = False`,
`error = None`, `store = False`, `store_id = 0`, `_requested = False`,
`num_borrowers = 0`, `parent = None`, `borrow_depth = 0`. `__getstate__`/`__setstate__`
unchanged (a pickled Future still works for store futures and for futures nested in
containers). `CollectiveFuture.send` keeps using the proxy entry method
`_coll_future_deposit_result`.

`EntryMethodThreadManager`: `start()` unchanged; `_resumeThread(gr, arg)`:

```python
if gr.notify: ...threadResumed...
obj = gr.obj
if _ckdyn.resume(gr, arg):
    obj._numthreads -= 1
    if obj._numthreads == 0 and getattr(obj, 'migratable', False):
        _ckdyn.set_migratable(obj, True)
```

`pauseThread` unchanged (`main_gr` is `getcurrent()` at `start()`, which equals
`_ckdyn.main_greenlet()`). `throwNotThreadedError` names
`charm.entryMethods[_ckdyn.last_ep()]` when `charm.last_em_exec` is None or stale
(use `_ckdyn.last_ep()` first; fall back to `last_em_exec`). `createFuture` uses
`FIDMAXVAL = 2**32 - 1` (already). `allreduce` in chare.py drops the `% 65535`.

### 3.5 `chare.py`

- `MAINCHARE, GROUP, ARRAY, CHARE = range(4)`; `CHARM_TYPES = (MAINCHARE, GROUP, ARRAY, CHARE)`.
  `charm.register(C)` registers GROUP, ARRAY and CHARE by default; `ArrayMap`
  subclasses GROUP only; the main chare MAINCHARE only.
- `Chare.__new__(cls, chare_type=None, args=[], onPE=-1, migratable=False)`: with a
  `chare_type` and `migratable=False`: `charm.proxyClasses[CHARE][chare_type].ckNew(args, onPE)`;
  with `migratable=True`: today's one-element-array path (`Array(..., ndims=1)`,
  `ckInsert(0, args, onPE, single=True)`). Default `__new__` otherwise.
- `Chare.__init__`: drop `_local`; keep `_active_grp_conds`, `_cond_next`,
  `_cond_last`, `_numthreads = 0`.
- `contribute`, `reduce`, `allreduce`: as today, through `charm.contribute`.
  `AtSync` → `_ckdyn.at_sync(self)`; `migrate(pe)` → raise `Charm4PyError` if
  `self._numthreads > 0`, else `_ckdyn.migrate(self, pe)`; `setMigratable` →
  `_ckdyn.set_migratable`; `__getRedNo__` → `_ckdyn.reduction_number(self)`.
  `_getSectionLocations_` deleted. Channel methods unchanged.
- Proxy generation: every generated proxy class inherits `_ckdyn.Proxy`. The
  per-method functions come from `_ckdyn.make_proxy_method(em.epIdx, argcount,
  argnames, defaults, em.name, proxyClassName + '.' + em.name)`; profiling mode wraps
  nothing (the C++ send path records nothing; `printStats` loses the "send" column
  for now, note it in the docstring). `__init__` of a group proxy class:
  `_ckdyn.Proxy.__init__(self, CKDYN_GROUP or NODEGROUP or ARRAYMAP, gid, 0)`; array:
  `(CKDYN_ARRAY, aid, ndims)`; mainchare/chare classes use `Proxy._from_target`.
  `__getitem__` for groups/arrays: copy and `_set_elem`/`_set_index`; a slice →
  `NotImplementedError('sections: step 4b')`. `ckNew`, `ckInsert`, `ckDoneInserting`,
  `ckLocalBranch` as today over `_ckdyn.create_*`. `updateGlobals`'s special proxy
  method (chare.py:502-542) stays a Python function (it rewrites its arguments) that
  calls `_ckdyn.send`. `__getstate__`/`__setstate__`/`__eq__`/`__hash__` come from
  `_ckdyn.Proxy`; the generated classes may add nothing.
- Creation flags: `Group.ckNew(args, onPEs)`: if called from a coroutine and the
  class is not an ArrayMap, `future = charm.Future()`, flags `BLOCK|BCAST|CREATION`,
  and `proxy.creation_future = future`; `onPEs` passes the PE list. `Array.ckNew`:
  same when `sum(dims) > 0`; `ckInsert(..., single=True)` from a coroutine: flags
  `BLOCK|PYMSG_SINGLE` and the future (BLOCK makes the constructor's slow path send
  `None` to it when `__init__` returns). `Chare` kind `ckNew(args, onPE)`: from a
  coroutine, flags `BLOCK|CREATION` with a future (CkDyn acknowledges it).
- `Mainchare.initMember(obj, target)` builds the proxy with `Proxy._from_target`;
  `Group.initMember(obj, gid)`, `Array.initMember(obj, aid, index, single)` as today
  minus `_contributeInfo`.

### 3.6 `reduction.py`

`ReducerContainer`: unchanged API; `addReducer` assigns `func._rid` (0, 1, 2, ... in
registration order) and records `self._by_id[rid] = func`. `nop` is a plain sentinel
object. `ReductionManager`: `populateConversionTables` builds `red_table[op][elem]`
from `_ckdyn.reducers()` (name → op and elem; `sum_int` is op SUM elem I32; the
`logical_*_bool` entries op AND/OR/XOR elem BOOL; when two names share an elem,
the first wins), `numpy_type_map`/`array_type_map`/`python_type_map` now map to
`CKDYN_ELEM_*` values (`int` → I64, `float` → F64, `bool` → BOOL). `prepare(data,
reducer, contributor)` returns `(rtype, data)`: nop → `(nop_type, None)`; a built-in
op with a matching type → `(type, buffer)` where a Python scalar or list is converted
to a numpy array of the elem's dtype (or `array.array` when numpy is absent); a
Python reducer → `(foreign_type, (func._rid, preprocessed_data))`.
`reducers_by_id` property for `charm._reduce_foreign`.

### 3.7 `__init__.py`

Unchanged exports. `charm4py.threads.Future` and `charm4py.wait.parse_cond_str`
remain importable.

## 4. Build

`pyproject.toml`: build backend `scikit_build_core.build`; `build-system.requires =
["scikit-build-core>=0.10", "nanobind>=2.2", "numpy"]`; runtime dependencies
unchanged (`numpy`, `greenlet`); `[tool.scikit-build] wheel.packages = ["charm4py",
"charmrun"]`, `editable.mode = "redirect"`. Version: static in `pyproject.toml`
(`2.0.0.dev0`) and `charm4py/_version.py` written from it.

`CMakeLists.txt` at the repository root:

1. Locate libcharm: `CHARM4PY_CHARM_BUILD_DIR` (a directory with `lib/libcharm.*`
   and `include/ckdyn.h`) must be set; if not, and `CHARM4PY_CHARM_DIR` is set, run
   `./build charm4py $CHARM4PY_BUILD_TRIPLET -j$N --with-production --disable-fortran
   --with-refnum-type="unsigned int"` there (defer this branch to step 6 if the
   `./build` option does not exist yet: then fail with a message saying to build
   Charm++ first and set the variable). Check `CkDynRefnumBytes() == 4` and
   `CkDynApiVersion() == 1` at import, in `init`.
2. `find_package(Python COMPONENTS Interpreter Development.Module)`, `find_package(nanobind)`
   via `python -m nanobind --cmake_dir`, `nanobind_add_module(_ckdyn NB_STATIC src/ckdyn_py.cpp src/objstore_py.cpp)`,
   include `<build>/include` and greenlet's include directory (`python -c "import greenlet, os; print(os.path.dirname(greenlet.__file__))"`
   contains `greenlet.h` under `greenlet/`), link `libcharm` from `<build>/lib`.
3. Copy `libcharm`, `libreconverse`, `liblci`, `liblct` (and on Linux every library
   `libcharm.so` needs from the build) into `charm4py/.libs` with the renaming and
   `install_name_tool`/`patchelf` fix-ups that `setup.py` lines 146-294 do today
   (port that code to a Python script `tools/bundle_libs.py` that CMake runs at
   build time; keep its logic, including the `reconverse` marker file that
   `charmrun/start.py` reads). Set the module's rpath to `@loader_path/.libs` on
   macOS and `$ORIGIN/.libs` on Linux, and install `.libs` next to the package.
4. `pip install --no-build-isolation -e charm4py` in the project venv with the
   three `CHARM4PY_*` variables of the project CLAUDE.md must work, and so must a
   non-editable `pip install ./charm4py`.

## 5. Gate (DESIGN.md step 4a)

Tests that must pass with 1, 2 and 4 processes (`python -m charmrun.start python
<test> +pN +libcharm_interface cython`, or the per-test runner from
`logs/`): `tests/entry_methods/*`, `tests/collections/*`, `tests/futures/*`,
`tests/reductions/*` except `section_reduction.py`, `tests/thread_entry_methods/*`,
`tests/when/*`, `tests/callbacks/*`, `tests/qd/qd.py`, `tests/dcopy/test_dcopy.py`,
`tests/channels/*`, `tests/exceptions/test.py`, `tests/topo/*`,
`tests/array_maps/test1.py`, `tests/charm_remote.py`, plus
`benchmarks/pingpong.py` (branch `thin-bench` has the `-n BYTES` option) at or
below DESIGN.md 7.4's targets: plain 1.0 us, coroutine 1.4 us, 1 MB about 23 us.
Not in the 4a gate: `pool/*`, `sections/*`, `migration/*`, `interactive`, examples,
`appspector`.

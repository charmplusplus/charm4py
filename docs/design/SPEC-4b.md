# Step 4b specification: sections and constrained groups

Written 2026-11-08 with the implementation (branch `thin`). It extends
SPEC-4a.md (whose conventions hold unchanged) with sections, on top of the CkDyn
section API of `ckdyn-dev` c81fbeecc (`ckdyn.h` "Sections"; the internals chapter
`doc/internals/ckdyn.md` "Sections"). DESIGN.md 3.9 and 5.2 are the design; the
differences from 5.2 are listed in section 6.

## 1. What changed for users

- `charm.split`, `charm.combine`, proxy slicing (`g[::2]`, `a[0:8:2, 1:8:2]`) and
  `Group(C, onPEs=[...])` work again. They must be called from a coroutine (every
  test and example does: `main` is one); called elsewhere they raise
  `Charm4PyError`. They return when every member has joined the new section(s),
  so the returned proxies can be used at once.
- `charm.combine` accepts arrays and array sections (combined sections included)
  of one chare class with one index dimension count, and raises `TypeError` for
  anything else, group sections in particular (DESIGN.md 5.4 item 15). The result
  is a union: an element in two parts is a member once.
- Section members may migrate (DESIGN.md 5.4 item 10; CkMulticast rebuilds its
  tree). Sections are not checkpointed: after `+restart` the program creates its
  sections again.
- `proxy.section` is `(root_pe, sid)`, where `sid` is a 16-byte `bytes` object
  (the `CkDynSectionID`), usable as a dictionary key and as the `section=`
  argument of `contribute`/`reduce`/`allreduce`.
- In a constrained group, `thisProxy` is the plain group proxy while the user
  constructor runs and becomes the group-section proxy right after it (when the
  branch joins the section), before any message from the creator can arrive.

## 2. The binding (`src/ckdyn_py.cpp`)

**Section proxies.** `ProxyObject` gains `issec` and a `CkDynSectionID sid`.
`proxy._as_section(sid)` returns a new proxy of the same class, without an
element, with `issec` set. `issec` and `section` read the fields;
`__getstate__` appends the sid bytes for a section proxy (and `__setstate__`
accepts both forms); equality and hash of section proxies compare the sid only.
A section proxy passed as an argument is pickled (SPEC-4a section 1, last row);
the PROXY slot carries collection and element proxies only.

**Sends.** A proxy method called on a section proxy builds the payload as for any
send, finishes it with `CkDynMsgFinishMcast` and sends it with
`CkDynSendSection(sid, ep, m, NULL)` (from a PE other than the root, CkDyn
forwards it to the root). `CKDYN_MSG_BCAST` is set only together with BLOCK (a
`ret=True` or `awaitable=True` call, whose members contribute to the future over
the section), so a plain section send has header flags 0 and reaches the
receivers' fast path.

**Delivery.** An MCAST view is converted as a DYN view, with read-only numpy
buffers (DESIGN.md 5.4 item 8). The slow-path header gets `b'sid'` (the 16 bytes
of `view.section`, copied during the hook), used by `EntryMethod._run` to
contribute a `ret`/`awaitable` result over the section, and `b'constrained'` for
a constructor message with `CKDYN_MSG_CONSTRAINED`.

**Reduction results at a section entry** (a SECTION callback). CkDyn relays a
built-in result as an MCAST payload with one BUFFER slot of `ndim` 0 and no
header flags, which a user send of a 0-dimensional numpy array would also
produce. The binding therefore tags a 0-dimensional numpy argument
`TAG_ND0` (= 4) when it sends one, and a MCAST view whose only slot is a BUFFER
of `ndim` 0, tag 0 and no flags is converted as a REDRESULT view (SPEC-4a
section 1: `None`/no argument for an empty result, a scalar for one element,
else a numpy array). A foreign result keeps its payload and the
`PYMSG_REDRESULT` flag, as before. A non-zero `view.refnum` on an MCAST view is
inserted as the first argument and as `header[b'refnum']`, as for other
callback deliveries (SPEC-4a 2.3).

**Callbacks.** A bound proxy method of a section proxy converts to
`CKDYN_CB_SECTION` (`sid`, `ep`; `refnum` from `Callback.target(bpm, refnum)`);
`contribute`, `contribute_section`, `start_qd` and `checkpoint` accept it.
`Callback.sid` reads it back.

**Hook.** `section_joined(bd, handle, sid)` calls
`charm._section_joined(obj, sid_bytes)`.

**Module functions** (all section ids are 16-byte `bytes`):

```
section_create_array(aid, ndims, indices, bfactor=0, done=None) -> sid   # indices: flat ints, nelems*ndims
section_create_group(gid, pes, bfactor=0, done=None) -> sid
section_combine(sids, done=None) -> sid
section_delete(sid)
section_redno(obj, sid) -> int            # CkDynSectionRedNo
section_info(sid) -> (root_pe, idx, coll_id, kind, gen)
contribute_section(obj, sid, data, reducer_type, target)   # data as for contribute
CB_SECTION
```

`done` is anything `contribute` accepts as a target (a Future, a bound proxy
method, a `Callback`, `None` for IGNORE). `bfactor` 0 is CkMulticast's default
branching factor.

## 3. Python: split and combine (`charm.py`, `chare.py`)

**`charm.split(proxy, numsections, section_func=None, elems=None, slicing=None)`.**
`proxy` is a group, array or section proxy without an element.

1. Membership. A whole group with `elems` needs no vote: the members are the
   listed PEs. Otherwise the caller sends `proxy._sectionVote_(numsections,
   section_func, slicing, elems, f, secproxy)` to the collection or section;
   `_sectionVote_` is a reserved `Chare` method, so it is an entry method of every
   registered class. Each member computes its section numbers (from
   `section_func(self)`, the slices, or membership in `elems`; with none of them,
   section 0) and contributes `(coll_id, thisIndex, sections)` with
   `Reducer.gather` to the future `f` (over the section when `proxy` is one).
2. Creation, on the caller's PE (the root of every new section). For each section
   number with its members: a group section with `section_create_group`; an array
   section with `section_create_array`; members from several arrays (a split of a
   combined section) become one array section per array, combined with
   `section_combine`. Every section's `done` is the one future
   `Future(numsections)`. A section without members raises `Charm4PyError`.
3. Readiness. `split` waits on that future, then returns
   `[proxy._as_section(sid), ...]`.

Slicing on a proxy calls `split`: `g[a:b:c]` with `elems=[range(a, b, c)]`
(defaults 0, `numPes()`, 1), `a[s0, s1, ...]` with `slicing`. Indexing a section
proxy with an index (not a slice) returns the element proxy of the underlying
group or array (a combined section raises `Charm4PyError`).

**`charm.combine(*proxies)`.** Checks first: every proxy is an array proxy or an
array-section proxy without an element, all of one proxy class (one chare class)
and one `ndims`; otherwise `TypeError`. A whole array is turned into a section
with `split(p, 1)`. Then `section_combine(sids, done)`, wait on `done`, return
`proxies[0]._as_section(sid)`.

**Readiness decision.** Blocking in `split`/`combine` on the `done` callbacks,
and a `Charm4PyError` outside a coroutine. Every gate test calls them from a
coroutine, and the vote of step 1 needs a coroutine anyway. A section proxy
therefore exists only once its section is ready, so no member can receive a
user message before `section_joined`, and nothing is buffered in Python.

## 4. Python: section reductions and collective futures

`_section_joined(obj, sid)` adds `sid` to `obj._sections` (a set, pickled with
the object, so it survives migration).

`charm.contribute(data, reducer, target, chare, section=None)`: when `section`
is None and `chare.thisProxy.issec` (a constrained-group branch), the section
is `thisProxy`. With a section (a proxy or a sid), the contributor must have
joined it (`Charm4PyError` otherwise); the data are prepared as for any
contribution and passed to `_ckdyn.contribute_section`. `reduce` and
`contribute` on `Chare` pass `section` through unchanged.

`Chare.allreduce(data, reducer, section=None)` on a section: the shared id is
`refnum = section_redno(self, sid) + 1`, identical on every member. The member
creates a `CollectiveFuture` keyed by `((refnum, sid), self)` and contributes
with the callback `Callback.target(secproxy._coll_future_deposit_result,
refnum)`, a SECTION callback. CkDyn relays the result to every member with
`refnum` as the envelope reference number; the binding passes it first, and
`charm._invoke`, seeing `b'refnum'` and `b'sid'` for
`_coll_future_deposit_result`, replaces it with `(refnum, sid)`. The key keeps
section and whole-collection collective futures apart (an element can wait on
both with the same number).

## 5. Constrained groups

`Group(C, args, onPEs=pes)` (from a coroutine): `create_group(..., pes)`
(CkDyn CONSTRAINED: only the listed branches are constructed; the constructor
header carries `b'constrained'`, and `_construct` marks the object), then
`section_create_group(gid, pes, 0, done)` and a wait on `done`. When a marked
branch joins its first section (that one: nothing else can name the group
earlier), `_section_joined` sets `thisProxy` to the section proxy, so
broadcasts, `contribute` and `allreduce` on `thisProxy` reach the listed PEs
only. The proxy returned to the caller is the same section proxy. No
`creation_future` is set: the call returns after creation.

## 6. Differences from DESIGN.md 5.2, each kept

1. The vote entry is `_sectionVote_`, a reserved `Chare` method registered like
   `_coll_future_deposit_result`, not `__sectionVote__` registered by the binding:
   names with leading and trailing double underscores are never entry methods.
2. The vote gathers with the existing `Reducer.gather` instead of a new reducer.
3. A whole group split by `elems` skips the vote (the members are PEs).
4. `_ckdyn.section_create_array` and `section_create_group` replace one
   `section_create(proxy, indices_per_section, bfactor)`.
5. The split of a combined section makes one array section per array and
   combines them, because `CkDynSectionCreateArray` takes one array.
6. `combine` also accepts whole arrays, and the "same index type" rule is
   checked as one proxy class and one `ndims` in Python (CkDyn checks the
   dimension count again).
7. In a constrained group, `thisProxy` becomes the section proxy after the
   constructor, not during it (section 5).
8. Indexing a section proxy returns the element proxy of the underlying
   collection (tests/sections/constrained_groups.py does `g[pe]` on the
   returned section proxy), as in 1.x.

# SPEC-7-coro: migratable coroutine entry methods on ckcoro's state record

Step 7 item, written 2026-11-12 22:15 PST. Design only; no code yet. It builds on the
ckcoro infrastructure that Charm++ PR #4231 merged into `recharm` (merge commit
b5248ed8b; `origin/recharm` tip 964c6626f when this was written): `src/ck-core/ckcoro.h`,
`src/ck-core/ckcoro_core.h`, `doc/internals/coroutine-sdag.md`, the manual section
"Structured Control Flow: Coroutine Entry Methods". Layer 2 work goes on Charm++ branch
`ckdyn-dev`, binding work on charm4py branch `thin`. No pull requests.

It replaces the step 7 item "cooperative coroutine migration at `migrateAndWait`"
(DESIGN.md 4.5 item 7, 7.1 step 7), see open question Q9.

## 1. The problem

A `@coro` entry method of charm4py 2 runs on a pooled worker greenlet. The greenlet
holds a slice of the C stack and Python frame objects whose pointers are valid only in
this interpreter, so a chare whose coroutine is suspended (on a future, a channel,
`charm.wait`, `self.wait`, `allreduce().get()`, `awaitCreation`) cannot be packed.
What the thin binding does today (charm4py `thin` at b3c4a58):

| situation | code | behavior |
|---|---|---|
| a `@coro` entry is delivered | `h_deliver` (src/ckdyn_py.cpp:1760; fast path at 1667) calls `EntryMethod.run` → `_run_th` (entry_method.py) → `_ckdyn.spawn` (`py_spawn`, ckdyn_py.cpp:2586) | the body runs on a pooled greenlet; `obj._numthreads` counts running and suspended coroutines |
| the body blocks | `EntryMethodThreadManager.pauseThread` (threads.py:303-333) | `obj._nsuspended += 1`; on 0 → 1, if `obj.migratable`, `_ckdyn.set_migratable(obj, False)` (`CkDynSetMigratable` → `CkMigratable::setMigratable(false)`), so every load balancer leaves the element where it is; restored on 1 → 0 |
| `self.migrate(pe)` while suspended | `Chare.migrate` (chare.py:165) | raises `Charm4PyError: Cannot migrate <Class><index>: it has N suspended coroutine(s)` |
| `self.ckDestroy()` while suspended | `Chare.ckDestroy` (chare.py) | raises the same kind of error |
| AtSync, then the coroutine returns | `Chare.AtSync` → `CkDynAtSync`; `_resume_from_sync` (charm.py:605) sends `resumeFromSync` as a message | the element migrates normally (tests/thin/atsync_in_coroutine.py phase 1) |
| AtSync, then the coroutine suspends | as above | the element is non-migratable for that balancing step and stays on its PE (same test, phase 2) |
| the runtime packs it anyway with reason MIGRATE | `Charm._pack` (charm.py:533) returns `None` on the packing pass when `obj._nsuspended > 0`; `h_pack_size` (ckdyn_py.cpp:1910) turns that into `CKDYN_PACK_REFUSED` | CkDyn's `pupObj` calls `bindingError` and the program aborts (the backstop for a coroutine that suspends between AtSync and the balancer's decision) |
| a checkpoint | `Charm._pack` packs regardless; `Charm._unpack` (charm.py:567) resets `_numthreads` and `_nsuspended` to 0 on a restore | the suspended coroutines are not in the checkpoint; after `+restart` the program continues only through entry methods and the checkpoint's `resume` target (DESIGN.md 4.5 item 4) |

So a chare written as one coroutine loop (jacobi2d.py: channels, `charm.iwait`,
`allreduce(...).get()`) is never load balanced and cannot continue from a checkpoint
unless it is split into callback-style entry methods. ckcoro solved this for C++; this
document adapts its mechanism to Python.

## 2. ckcoro's mechanism, in the terms the binding needs

### 2.1 What is recorded

A coroutine frame is never packed. The chare's `ck::coro::state` member (`__coro`,
declared by the generated `X_CORO_CODE`) holds:

- `recs_`, one `record` per active coroutine: entry index `ep`; `target`, a `CkCallback`
  to the owner rebuilt by `self_target` on every unpack; `msg`, the entry's message
  kept for restarts; `pending` (a restart message is on its way); the restart `chain`
  (`ckcoro_core.h`), whose `sites` hold one site id per resumable-function depth (depth
  > 0 only through `CK_CALL`); the wait state (`waiting`, `atSite`, `wants` with an
  `arm` per alternative, `readyBufs`); the frame handles `root` and `h`, not pupped.
- `inbox_`, `buffer {ep, ref, msg}` in arrival order: deliveries to `[inbox]` entries no
  coroutine was waiting for, pupped with `CkPupMessage`. The reference number is the
  envelope's for a message entry, the first arithmetic parameter for a marshalled
  entry, the reduction message's through a reduction wrapper.

### 2.2 Waiting and delivery

`CK_WHEN(want, out...)` is `case __LINE__: note_site(...); co_await state.when(...)`:
take the first matching buffer from the inbox (`state::take`), else store the `wants`
and suspend. A delivery calls `push`: resume a waiting record one of whose arms it
satisfies (`try_arm`), else append to the inbox. `CK_WHEN_ALL` is one arm with several
wants, `CK_WHEN_ANY` several arms.

### 2.3 Pack, unpack, restart

`state::pup` writes the inbox and, per record, `ep`, `target`, `msg`, `chain.sites`;
packing aborts if a record is running or suspended at a `co_await` that is not a site.
On unpack, `send_restart` sends a copy of the record's message through `target` unless
a restart is already `pending`. The promise adopts the restarting record, and
`CK_MIGRATABLE` (`frame_guard` plus `switch (site) case 0:`) jumps to the recorded site;
a recorded site that is not a case lands on `default:` and aborts. Deliveries that
arrive before the restart message are buffered as usual.

### 2.4 Rules, `+ck_fakemigrate`, checkpoint

- Values live across a site are members of the chare (pupped); a local does not
  survive a restart.
- Restart rule (Ritvik's correction in the review of #4231): code inside
  `CK_MIGRATABLE` before the recorded wait is skipped on a restart, so its results must
  be members; code before `CK_MIGRATABLE` runs again on every restart.
- A chare may migrate or be checkpointed only while its coroutines are suspended at
  sites.
- `+ck_fakemigrate` (parsed in init.C into `_ckCoroFakeMigrate`) makes the first
  suspension at each site call `state::rebuild`: the state is packed into memory, the
  frames destroyed, the state unpacked, and the restarts sent. It does not move the chare
  through its own `pup` (review: a member left out of `pup` keeps its value under the
  flag and is lost in a real migration).
- Checkpoint: the same pup; `pup(p, owner)` rebuilds `target` so a restored chare with a
  new handle is reached. Review findings that apply to any reuse: a restart message in
  flight is lost by a checkpoint unless `pending` is cleared when unpacking with
  `p.isCheckpoint()`; site ids are `__LINE__`, so a checkpoint restarts only with a
  binary built from the same source layout.
- One active coroutine per entry per chare (`begin` aborts on a second).

### 2.5 What a C API can reuse

| part of ckcoro | reusable by CkDyn | why |
|---|---|---|
| the record's persistent fields (`ep`, `target`, `msg`, `pending`, `chain.sites`) | yes, as data layout and protocol | plain data; `pup` is ordinary PUP code |
| the inbox (`buffer`, `take` in arrival order, matching by `(ep, ref)`, `try_arm` for all/any) | yes | no coroutine types involved |
| `send_restart`, `restart_copy`, `self_target` per chare kind | yes, as protocol; CkDyn already builds targets for every kind | CkDyn's targets replace `self_target` |
| the `pending` rule and the checkpoint fix | yes | |
| `+ck_fakemigrate` flag (`_ckCoroFakeMigrate`) | yes, through an accessor | a global in ck.C |
| `CK_MIGRATABLE`, `CK_WHEN*`, `CK_CALL`, `frame_guard`, `note_site` by `__LINE__` | no | C++ macros and `switch` jumps |
| `ck::coroutine` and its promise, `task<T>`, `when_awaiter`, `std::coroutine_handle` | no | C++20 coroutine machinery |
| the typed unpacking into outs (`want_t<params<...>>`, `array_arg`) | no | charmxi-generated C++ types; CkDyn delivers views |

`ckcoro.h` requires C++20 and is header-only; `libck` builds with `CMAKE_CXX_STANDARD
11`, so `ckdyn.C` cannot include it. Section 4 therefore specifies a CkDyn-side record
and inbox that follow ckcoro's protocols (about 300 lines of C++11 in `ckdyn.C`), not a
wrapper around `ck::coro::state`. Factoring the inbox and record into a shared C++11
header used by both is possible afterwards; it is not required (Q3).

## 3. The Python contract

### 3.1 The form

```python
from charm4py import Chare, coro, inbox

class Jacobi(Chare):
    @inbox
    def ghost(self, it, direction, data):   # never runs: deliveries go to the inbox
        pass

    @coro(migratable=True)
    async def run(self, niter):
        while self.it < niter:
            self.send_ghosts()
            for self.k in range(self.nnbrs):
                it, d, data = await self.inbox.ghost(self.it)
                self.set_ghost(d, data)
            ...
```

- A migratable coroutine entry is an `async def` method decorated
  `@coro(migratable=True)`. `@coro` without arguments stays the greenlet form (3.8).
  `async def` is required because `await` marks every wait syntactically, which makes
  the sites enumerable at decoration time, and because a native coroutine object can be
  driven by the binding with `send()` without a greenlet.
- `@inbox` declares an inbox entry, the counterpart of ckcoro's `[inbox]`. Its
  deliveries are buffered by CkDyn in the chare's inbox (section 4), matched by entry and
  reference number, and move with the chare. `self.inbox.<name>(ref)` names a want
  (ckcoro's `inbox::name(ref)`); `self.inbox.<name>()` matches any reference number.
  The reference number of a delivery: the envelope's when non-zero (reduction results,
  element futures, sends with an explicit `ref=`), otherwise the first positional
  argument when it is an `int`, otherwise none.
- The awaited value is the delivery's arguments: `None` for none, the value for one,
  a tuple for several.

### 3.2 Sites

A site is an `await` that is a statement of the entry's own body: `await w` or
`target = await w` (any assignment target). The transform (3.4) numbers sites 1, 2, ...
in source order; 0 means "start". Compared with ckcoro's `__LINE__`, ordinals do not
change when lines are added, comments edited or code moved without reordering waits, so
a checkpoint survives ordinary edits; adding, removing or reordering a site, or moving
it into a different `if`/`while`, changes the numbering. The record therefore also
stores a 64-bit *layout* value: a hash of the site skeleton (for each site, the chain of
enclosing compound statements and their site sets, without line numbers or the
statements between sites). A restart whose layout differs aborts before any statement
runs (5.1). An explicit `site=` name per wait is not proposed (Q8).

An `await` anywhere else is rejected when the class is registered, with the line:
inside an expression (`f(await w)`), a comprehension, a lambda or nested function, a
`try`, `with`, `async with`, `async for` or `match` statement, or a `for` loop whose
target is not an attribute of `self` or whose iterable is not `range(...)`, or a
`while`/`for` with an `else` clause. A charm4py waitable awaited outside a site (in a
helper `async def` called from the body) raises `Charm4PyError` at that `await`; the
waitable's `__await__` checks a flag that only the transform's site wrapper sets.
Helper coroutines with their own sites (ckcoro's `CK_CALL`, chain depth > 0) are left
for later (Q7).

### 3.3 Re-running the body to the recorded site: the options

ckcoro's switch jumps over the code before the recorded site. Python has no jump, and
the frame cannot be serialized. Four options:

| option | what runs on a restart | consequence |
|---|---|---|
| (a) re-run the plain body; each site below the recorded one returns at once | every statement between the start and the recorded site, again | sends and contributions are repeated (a duplicate ghost message or a second contribution to a reduction is a wrong answer or a hang); a site below the recorded one must return something, and the value it returned the first time was consumed, so either a value log grows with every iteration or the code after it sees a placeholder |
| (b) body as a generator driven by the binding | the same as (a) | removes the greenlet and marks waits with `yield`, but a generator cannot resume in the middle either; it does not solve the re-run |
| (c) the programmer writes a state machine (`if self.phase == 1: ...`) | only what the programmer guards | correct, but every program carries the bookkeeping that ckcoro's macros remove; the form SDAG and ckcoro were written to avoid |
| (d) the binding generates (c) from the body: a decoration-time AST transform inserts restart guards | nothing before the recorded site; code after it runs as usual | the same restart rule as ckcoro (code before the recorded site inside the body is skipped), no value log, no repeated sends |

**Recommendation: (d).** It is the only option that gives ckcoro's semantics without a
log: the transform is the Python equivalent of the `switch` that `CK_MIGRATABLE` opens,
generated rather than written. (a) is wrong for the main use case, an iteration that
sends before it waits: the restarted body would send the ghosts of the current
iteration a second time. (b) is adopted for the driving model (an `async def` driven by
`send()`, no greenlet), not as a restart method. (c) remains available: guards a
programmer writes are ordinary statements, which the transform leaves in place.

### 3.4 The guard transform

At class registration (`Charm.registerAs`, charm.py:783, where `entry_flags` is
computed), for each `@coro(migratable=True)` method, the binding reads the source (`inspect.getsource`),
parses it (`ast.parse`), checks the rules of 3.2 and 3.5, and compiles a transformed
function with the original file name and line numbers (`ast.copy_location` on every
generated node), so tracebacks, `pdb` and coverage point at the user's lines. The
transformed function takes a hidden argument `_rs`, the restart site (0 on a new
start). Rules, applied to every block that contains a site:

1. A maximal run of statements that contain no site becomes `if _rs == 0: <run>`.
2. A compound statement that contains sites S becomes `if _rs == 0 or _rs in S:`
   around its transformed form.
3. `if c: A elif d: B else: C` becomes `if _rs in S(A) or (_rs == 0 and c): A' elif
   _rs in S(B) or (_rs == 0 and d): B' else: C'`. The test is not evaluated when
   restarting into a branch, as the `switch` does not evaluate it.
4. `while c: B` becomes `while _rs in S(B) or (_rs == 0 and c): B'`.
5. `for self.x in range(lo, hi, st): B` (loop number n) becomes: under rule 1,
   `self.x = lo` and `self._ckcoro_range[n] = (hi, st)` (the bounds evaluated once, as
   Python does, and kept in the chare); then `while _rs in S(B) or (_rs == 0 and
   self.x < hi)` (`>` for a negative `st`), with `hi` and `st` read from
   `self._ckcoro_range[n]`, around `B'; self.x += st`; every `continue` of that loop
   becomes `self.x += st; continue`.
6. A site `await w` or `T = await w` (site k) becomes

   ```python
   if _rs == k:
       _rs = 0
       T = await _ck_rewait(k)
   else:
       T = await _ck_site(k, w)
   ```

   On a restart the operand `w` is not evaluated again: it may have side effects
   (`self.allreduce(...)` contributes before it waits) and may read locals that the
   restart did not assign. `_ck_rewait` waits for the wants recorded at site k, which
   CkDyn keeps in the record (section 4). From here the body runs as usual.

Other statements are unchanged; a body without sites is rejected. On a new start each
guard costs one comparison of a local integer per run of statements. With `_rs == k`
the function evaluates only the guards of rules 2 to 5 on the path to site k and
reaches site k having run no user statement, not even the site's operand.
`_ck_site(k, w)` marks the waitable with its site; the driver (3.6) sees `(k, wants)`
when the coroutine yields.

### 3.5 State across a site

As in ckcoro, values live across a site are attributes of `self`, which the binding
pickles with the chare (`Charm._pack`). Locals are not rebuilt: on a restart the
statements that assigned them before the site are skipped. Parameters keep their
delivered values, because the record keeps the entry's message (section 4, retained,
not copied); numpy arguments of a migratable entry are therefore passed as read-only
views, so that the body cannot change what a restart will see.

The transform checks this at registration, which ckcoro cannot do in C++: a backward
liveness pass over the body's control flow (straight line, `if`, `while`, the
rewritten `for`, `break`, `continue`, `return`) finds every local that is assigned
before a site and may be read after it without being assigned again in between, and a
parameter that is reassigned before a site and read after it. Either is an error:

```
Charm4PyError: Jacobi.run (jacobi.py:74): local 'iteration' is assigned at line 74
and read at line 92, after the await at line 80; a value used across an await must be
an attribute of self (self.iteration)
```

The value a site returns may be a local: it is assigned after the wait and is live
only until the next site. Loop variables of a `for` loop that contains a site must be
attributes (rule 5). What the check cannot see (a local read through `locals()`, or
`exec`) fails on the restart with `UnboundLocalError`, which `+ck_fakemigrate` (5.3)
reaches in a 1-PE run.

Side effects: statements before the recorded site inside the body do not run again, so
no rule about repeated side effects is needed inside the body. What must run on the new
PE (re-posting a buffer with `post_buffer`, re-creating an object that `__getstate__`
dropped) goes in `__setstate__` or `migrated()`, as for any migratable chare; the
manual page says so. Argument conversion and the decorator machinery run again; they
have no effect visible to the program.

### 3.6 What can be awaited

The binding drives the coroutine object: `deliver` creates it with `_rs`, calls
`send(None)`, and on each yield of a site calls `CkDynCoroWait` (section 4). If a
buffered delivery satisfies the want, the driver converts it and sends it back in at
once; otherwise it returns, and the matching delivery later reaches `deliver` with
`CKDYN_DELIVER_RESUME`. No greenlet is involved, so the blocking greenlet calls
(`Future.get`, `Channel.recv`, `charm.wait`, `self.wait`, `charm.iwait`) raise
`Charm4PyError: Jacobi.run is a migratable coroutine: use 'await f' instead of
f.get()` when called from the body or from a function it calls (they reach
`pauseThread` on the main greenlet, whose error message is extended for this case).

| awaitable | kind | mechanism |
|---|---|---|
| `self.inbox.e(ref)`, `self.inbox.e()` | site | want `(ep of e, ref)`; CkDyn's inbox |
| `charm.all(w1, w2, ...)` | site | one arm with several wants (ckcoro `CK_WHEN_ALL`); value: a tuple of values |
| `charm.any(w1, charm.all(...), ...)` | site | one arm per alternative (`CK_WHEN_ANY`); value: `(index, value)` |
| `self.allreduce(data, reducer)`, `self.reduce(data=..., reducer=...)` without a callback, called in a migratable body | site | an element future (3.7) as the reduction callback: `TARGET` broadcast to the hidden inbox entry `__ckred__`, reference number = reduction number + 1, shared by all contributors as today (chare.py `allreduce`) |
| `self.future()` | site | an element future (3.7) |
| `charm.qd()` | site | `CkDynStartQD` with an element future's `TARGET` callback |
| `self.AtSyncWait()` | site | `CkDynAtSync`, then a want on the hidden inbox entry `__ckresume__`, to which `_resume_from_sync` sends instead of `resumeFromSync` while a migratable coroutine waits there |
| `charm.checkpoint_wait(dir)` | site | `CkDynCheckpoint` with an element future's `TARGET` callback; resumes in the original run and, after `+restart`, in the restored one (5.1) |
| a PE future (`Future()`, `proxy.m(..., awaitable=True)`, `ret=True`) | pinning wait | not a site: the driver counts it in `obj._nsuspended` and calls `set_migratable(obj, False)` as `pauseThread` does today; packing while there is refused (1); `+ck_fakemigrate` does not rebuild there |
| `ch.recv()` on a `Channel` | pinning wait (phase 1); site later (Q6) | channel messages arrive at `_channelRecv__` with a sequence number; an inbox form needs a reference number that encodes port and sequence number |
| `charm.awaitCreation(...)`, `charm.split`/`combine` | pinning wait | creation acknowledgements and section joins go to PE futures |
| anything else (an asyncio awaitable) | error | `_ck_site` accepts only charm4py waitables |

The binding knows the kind from the waitable's class: a want, `all`/`any` and element
futures are sites; PE futures and channels pin. A pinning wait is allowed so that a body
can still call a method with `ret=True`; the element is then non-migratable only for the
duration of that wait, which is today's rule applied to one wait instead of the whole
coroutine.

### 3.7 Element futures: awaiting a future or callback directly

Ritvik's follow-up for #4231 ("awaiting a callback or future directly, for example a
reduction result or quiescence; charm4py awaits these directly") names exactly what a
migratable wait on a future needs. A charm4py future today is PE-pinned: its callback
is `CKDYN_CB_FUTURE (pe, id)`, the value goes to `CkDynMgr[pe].deposit` and the `future`
hook resumes the greenlet on that PE (ckdyn.md "Callbacks and futures"). After the
waiting chare migrates the value still goes to the old PE, and after a restart the id
is stale. That is why such a wait pins.

An *element future* is addressed to the chare instead of the PE: its callback is
`CKDYN_CB_TARGET` to the chare's hidden inbox entry `__ckfut__` with `refnum` = an id
from a per-chare counter (kept in the chare's pickled state, so ids never repeat across
migrations and restarts). Every producer already exists in CkDyn: `CkDynCallbackSend`
with a `TARGET` callback and `refnum` (for `f.send(v)` from any PE), `CkDynContribute`
with that callback (reductions), `CkDynStartQD`, `CkDynCheckpoint`. The location manager
forwards the value to wherever the element lives, CkDyn buffers it in the inbox when the
coroutine is not yet waiting, and both move with the chare. For a group branch the
target is the branch on its PE, and for a main chare CkDyn routes `TARGET` callbacks
through `CkDynMgr`'s relay, which also survives a checkpoint (ckdyn.md). An element
future pickles as `(target, id)` and can be passed in messages like a PE future.

So what CkDyn needs for a future or callback to be an inbox-backed site is not a new
callback kind but the inbox itself (section 4), plus `CkDynViewEp`, which tells the
binding which inbox entry a reduction or system message was delivered to (such a view
has no `CkDynHdr`). In C++ the same construction is an `[inbox, reductiontarget]` entry
declared by the runtime instead of the user.

### 3.8 Greenlet `@coro`, AtSync, and other entries

- `@coro` (greenlet) entries stay as they are: the general form, any call depth, not
  migratable while suspended. A chare may have both kinds; a suspended greenlet
  coroutine still makes the whole chare non-migratable.
- An `async def` method without `@coro(migratable=True)` is not an entry method (a
  proxy call of it raises `Charm4PyError`); it may serve as a helper without sites (Q2).
- AtSync inside a migratable body: `await self.AtSyncWait()` is a site, so the element
  is migratable while it waits, and the body continues on the new PE after
  `ResumeFromSync`. `self.AtSync()` followed by a return (today's pattern) keeps working.
- One active invocation per migratable entry per chare, as in ckcoro: a second delivery
  while a record of that entry exists raises `Charm4PyError` (Q10).
- A migratable entry called with `ret=True`/`awaitable=True`: the block future is in
  the kept message, so the return value is sent after a migration as well. After a
  checkpoint restart the caller's PE future belongs to the old run and the send raises
  the existing "future from before the restart" error (DESIGN.md 4.6).
- `@entry(inline=True)` on a migratable entry is a registration error, as for `@coro`.

## 4. CkDyn additions

All are additions within `CKDYN_API_VERSION 2`: new enum values, new functions, new
flag bits in the existing `ep_flags` argument of `deliver`. No hook is added and no
struct grows, so `sizeof(CkDynHooks)` and the `static_assert`ed struct sizes do not
change, and a binding built against version 2 without them still loads.

```c
/* Entry attributes (added within version 2). */
enum {
  CKDYN_EP_CORO  = 1u << 9,   /* a migratable coroutine entry: CkDyn keeps its record (CkDynCoro*) */
  CKDYN_EP_INBOX = 1u << 10   /* deliveries are buffered in the object's inbox; the binding sees
                                 one only through CKDYN_DELIVER_RESUME or CkDynCoroReady.
                                 Registered trace-disabled (CK_EP_TRACEDISABLE) */
};

/* Bits CkDyn sets in the ep_flags argument of hooks.deliver (added within version 2). */
#define CKDYN_DELIVER_RESTART (1u << 30)  /* msg is the kept message of a CORO entry, sent again after
                                             an unpack or a rebuild; CkDynCoroBegin returns the site */
#define CKDYN_DELIVER_RESUME  (1u << 31)  /* a delivery satisfied an arm of the waiting coroutine of
                                             entry ep (a CORO entry); read the arm's messages with
                                             CkDynCoroReady. msg is the delivery that completed the arm */

typedef struct CkDynWant {
  int32_t  ep;        /* an INBOX entry of the object's class */
  uint32_t ref;       /* reference number to match; ignored when any_ref != 0 */
  uint8_t  any_ref;
  uint8_t  arm;       /* the alternative this want belongs to: 0 for a single want or an all */
  uint16_t tag;       /* binding-defined, kept and pupped with the want (charm4py: the shape
                         of the awaited value: one want, an all, or an any) */
} CkDynWant;          /* 12 bytes; static_assert in ckdyn.C */

/* Called by the binding in hooks.deliver for a CORO entry, before it converts the
 * arguments. A new start (no record of ep): CkDyn creates the record and keeps the
 * message (CkDynMsgRetain semantics; no copy), returns 0. With
 * CKDYN_DELIVER_RESTART: CkDyn adopts the record, checks that layout equals the
 * recorded one (aborts otherwise, naming the class, the entry and both values), and
 * returns the recorded site (>= 1). A second start while a record of ep exists aborts
 * through hooks.error ("<Class.entry> is already running on this object"). */
int  CkDynCoroBegin(const CkDynTarget *t, int ep, uint64_t layout, const CkDynMsgView *msg);

/* The coroutine of entry ep is at site `site` (>= 1) and waits for one arm of
 * wants[0..n-1]. Records the site and the wants. On a restart the binding passes
 * wants == NULL: CkDyn checks that site is the recorded one and uses the recorded
 * wants (the operand of the await is not evaluated again, 3.4). Returns the index of an arm that buffered deliveries satisfy (taken from
 * the inbox in arrival order; read them with CkDynCoroReady), or -1: the record now
 * waits, and the delivery that completes an arm arrives through hooks.deliver with
 * CKDYN_DELIVER_RESUME. +ck_fakemigrate is the binding's business (CkDynCoroRebuild). */
int  CkDynCoroWait(const CkDynTarget *t, int ep, int32_t site, const CkDynWant *wants, int n);

/* The messages of the satisfied arm, in want order: fn is called once per message
 * with a view that is valid during the call (CkDynMsgRetain keeps the message, as in
 * hooks.deliver); *arm receives the arm index. The messages are released after the
 * last call. Returns the number of messages. */
typedef void (*CkDynViewFn)(void *arg, const CkDynMsgView *v);
int  CkDynCoroReady(const CkDynTarget *t, int ep, int *arm, CkDynViewFn fn, void *arg);

/* The body of entry ep returned or raised: drop its record and release its message. */
void CkDynCoroEnd(const CkDynTarget *t, int ep);

/* What a migration does to the object's coroutine records, in place: pack the
 * records and the inbox into memory, unpack them, and send the restarts. The binding
 * calls it after it has dropped its coroutine objects (and, for charm4py, round-tripped
 * the chare through pickle, 5.3). */
void CkDynCoroRebuild(const CkDynTarget *t);

/* +ck_fakemigrate was given (_ckCoroFakeMigrate, init.C). */
int  CkDynCoroFakeMigrate(void);

/* Active records of the object, and the buffered inbox deliveries (diagnostics, tests). */
int  CkDynCoroCount(const CkDynTarget *t);
int  CkDynCoroInboxCount(const CkDynTarget *t);

/* The entry index a delivered message was sent to (the envelope's), for REDRESULT and
 * SYS views, which have no CkDynHdr. */
int  CkDynViewEp(const CkDynMsgView *v);

/* Set the envelope's reference number of a finished message (an inbox send with
 * ref=). The same checks as future ids: not 0, not all-ones. */
void CkDynMsgSetRefnum(CkDynMsg *m, uint32_t ref);
```

Inside CkDyn (`ckdyn-impl.h`, `ckdyn.C`):

| piece | where | relation to ckcoro |
|---|---|---|
| `CkDynCoroState` on `CkDynObj` (allocated at the first CORO or INBOX delivery): `std::vector<Rec>` and `std::vector<Buf>`; `Rec {ep, msg, pending, site, layout, waiting, wants, ready, readyArm}`; `Buf {ep, ref, msg}` | new, C++11 | the persistent fields of `record` and `buffer` (2.1); no frame handles, no chain beyond one depth |
| `dispatch` for an INBOX ep: compute `ref` (envelope's if non-zero, else the first positional I64/U64 slot after a BLOCK future slot, else none), keep the message, try the waiting records' arms in record order (`try_arm`), else append | new | `state::push`, `try_arm`; the reference rule of `deliver_message` |
| the resume call: `CkDynCoroResume(t, coro_ep, rec id)`, `hooks.deliver(..., coro_ep, ..., flags \| CKDYN_DELIVER_RESUME, view)`, `CkDynCoroSuspend` | new | fixes, for CkDyn, the Projections gap (5.2) |
| `pupObj`: after the binding's bytes, pup the state (records: `ep`, `pending`, `site`, `layout`, `wants`, `msg` with `CkPupMessage`; inbox) | extension of `pupObj` | `state::pup`; `pending` cleared when `p.isCheckpoint()` (review fix) |
| packing a record that is not waiting at a site (running, or at a pinning wait) | `pupObj` | `bindingError("<Class.entry> of <object> is suspended at a wait that is not a migration site")`, as ckcoro aborts; charm4py refuses earlier through `CKDYN_PACK_REFUSED` |
| after `unpack` returns the handle: for each record not `pending`, send a copy of `msg` to the object's target and entry `ep` (`CkDynSendTarget`; for a main chare through the relay), set `pending` | new | `send_restart`; the target comes from `pupObj`'s argument, so a restored chare with a new handle and a group branch on its PE are reached (`pup(p, owner)`) |
| `CkDynCoroRebuild` | new | `state::rebuild` |
| `CkDynCoroFakeMigrate` | accessor | `_ckCoroFakeMigrate` |
| an INBOX or CORO message with NCPY or DEVICE slots (`nbuf > 0`) | `dispatch` aborts | ckcoro keeps messages; a zero-copy slot's data does not survive a pack. The charm4py sender never uses zero-copy for these entries (it knows their flags from registration) |
| `CKDYN_EP_CORO` on a nodegroup | registration error in phase 1 | ckcoro makes such entries exclusive; nodegroups wait for multi-PE processes (DESIGN.md 8.3) |
| destruction of an object with records or buffered deliveries | release the messages | |

The C++ test binding (`tests/charm++/ckdyn`) gets a phase `coro` (8, step 1).

## 5. Checkpoint and restart, Projections, `+ck_fakemigrate`

### 5.1 Checkpoint and restart

A chare whose migratable coroutines all wait at sites is checkpointed like any chare:
`Charm._pack` pickles the Python state (including `_ckcoro_range` and the element-future
counter), CkDyn pups the records and the inbox after it. Records `pending` at the time
are written as not pending (the restart message in flight is not in the checkpoint) and
restart on restore. On `+restart`, the restore unpacks each chare and CkDyn sends the
restarts; each body runs to its recorded site and waits; the program continues. A
coroutine waiting on `charm.checkpoint_wait(dir)` is resumed by the checkpoint callback
in both runs, so `main` can be a migratable coroutine of the main chare that continues
after a restart (Q5), which removes the limitation of DESIGN.md 4.5 item 4 for programs
written that way.

Restart with changed source: the layout check (3.2) aborts with `CkDyn: restart of
Jacobi.run: the layout of its awaits differs from the checkpoint (recorded
0x..., this program 0x...)`. Edits that do not touch the site skeleton (statements
between sites, comments, new methods) restart normally, which answers the review's
note that line-number sites tie a checkpoint to the source layout.

Element futures resume after a restart; PE futures from the old run do not (existing
rule). A chare at a pinning wait cannot be checkpointed: packing aborts as in section 4.
(A suspended greenlet is checkpointed today because its coroutine is dropped; a
migratable record cannot be dropped silently.) See Q4.

### 5.2 Projections

A resumed segment is traced as an execute event of the coroutine entry
(`Class.method`), not of the inbox entry that resumed it: inbox entries are registered
trace-disabled, and CkDyn brackets the resume call with `CkDynCoroResume(t, coro_ep,
id)`/`CkDynCoroSuspend(t)` (ckdyn.md "Coroutines: load and trace attribution"), which
also charges the time to the element for load balancing. The first segment and a
restart are deliveries of the coroutine entry and traced as such. This is the event per
resumed segment the review of #4231 asks for in C++, where a segment is traced under
the inbox entry that resumed it (after `ResumeFromSync`, under the balancer's call); the
same bracket in `state::push` would close it there (Q11). A pinning wait is resumed from
the `future` hook, which the binding already brackets (DESIGN.md 3.13).

### 5.3 `+ck_fakemigrate`

The flag is parsed by Charm++ (init.C) and read with `CkDynCoroFakeMigrate`. When a
body first waits at a site it was not just restarted to and the wait does not complete
at once, the driver:

1. closes the Python coroutine object (`close()` raises `GeneratorExit` at the `await`;
   no user code runs, since sites are not inside `try` or `with`);
2. round-trips the chare through its own serialization:
   `new = pickle.loads(pickle.dumps(obj)); obj.__dict__.clear();
   obj.__dict__.update(new.__dict__)`, the same `__getstate__`/`__setstate__` a
   migration runs, so an attribute that is not pickled is lost here as it would be in a
   migration (the gap the review found in ckcoro's flag, closed for Python);
3. calls `CkDynCoroRebuild(t)`, which sends the restart.

`migrated()` is not called (the chare did not move). A 1-PE run under the flag
therefore checks the restart path of every site, the liveness rule, and the pickling
of the chare. A real move at every site (`CkDynMigrateMe` to the next PE) is a further
mode for multi-PE tests (Q12).

## 6. Examples

### 6.1 Jacobi, before

examples/jacobi/jacobi2d.py `Jacobi.run` (greenlet form, abridged): locals across
waits, channels, `iwait`, `allreduce(...).get()`; never migrated while it runs.

```python
@coro
def run(self):
    iteration = 0
    converged = False
    while not converged and iteration < MAX_ITER:
        self.send_ghosts()                       # Channel.send to each neighbor
        for nb in charm.iwait(self.nbs):
            direction, ghosts = nb.recv()
            self.set_ghost(direction, ghosts)
        max_error = check_and_compute(...)
        self.temperature, self.new_temperature = self.new_temperature, self.temperature
        converged = self.allreduce(bool(max_error <= THRESHOLD), Reducer.logical_and).get()
        iteration += 1
    if self.thisIndex == (0, 0):
        self.sim_done_future.send(iteration)
```

### 6.2 Jacobi, after

```python
class Jacobi(Chare):
    def __init__(self, sim_done_future):
        ...                                          # as before, without Channels
        self.nbs = [(LEFT, self.thisProxy[(x-1, y)]), ...]   # (direction seen by the neighbor, proxy)
        self.iteration = 0
        self.converged = False

    @inbox
    def ghost(self, it, direction, ghosts):          # reference number: it (first int argument)
        pass

    @coro(migratable=True)
    async def run(self):
        while not self.converged and self.iteration < MAX_ITER:          # site set {1, 2, 3}
            for d, nb in self.nbs:                                       # no site: rule 1
                nb.ghost(self.iteration, d, self.face(d))
            for self.k in range(len(self.nbs)):                          # rule 5
                it, direction, ghosts = await self.inbox.ghost(self.iteration)   # site 1
                self.set_ghost(direction, ghosts)
            max_error = check_and_compute(...)                           # local, dead at site 2
            self.temperature, self.new_temperature = self.new_temperature, self.temperature
            self.converged = await self.allreduce(bool(max_error <= THRESHOLD),
                                                  Reducer.logical_and)   # site 2
            self.iteration += 1
            if self.iteration % LB_PERIOD == 0:
                await self.AtSyncWait()                                  # site 3
        if self.thisIndex == (0, 0):
            self.sim_done_future.send(self.iteration)
```

Ghosts of iteration i+1 that arrive while the element waits at site 2 of iteration i
are buffered under reference number i+1. If the element migrates while waiting at site 1
with `self.k == 2`, the restart enters the `while` and the `for` without evaluating their
tests or `self.k = 0`, skips the sends, and waits for the third ghost; the first two were
already applied to `self.temperature`, which moved with the chare.

### 6.3 A reduction wait with an explicit target

```python
@inbox
def total(self, s):
    pass

@coro(migratable=True)
async def step(self):
    ...
    self.contribute(self.local_sum, Reducer.sum, self.thisProxy.total, ref=self.iteration)
    self.global_sum = await self.inbox.total(self.iteration)
```

`ref=` sets the callback's reference number (new keyword of `contribute`, through
`CkDynCallback.refnum`). `await self.allreduce(...)` does the same with the hidden
`__ckred__` entry and is the recommended form.

### 6.4 Breaking a rule

| mistake | when | message |
|---|---|---|
| `iteration = 0` before a site, read after | registration | the liveness error of 3.5 |
| `x = f(await w)` | registration | `Charm4PyError: Jacobi.run (jacobi.py:81): an await in a migratable coroutine must be a statement 'await w' or 'target = await w'` |
| `for i in range(n):` around a site | registration | `... the target of a for loop that contains an await must be an attribute of self ('for self.i in range(...)')` |
| a site inside `try:` or `with` | registration | `... an await cannot be inside a try or with statement in a migratable coroutine (line 85)` |
| `await self.inbox.ghost(i)` inside a helper `async def` | first execution | `Charm4PyError: await of self.inbox.ghost outside the body of a migratable coroutine entry (Jacobi.recv_all, jacobi.py:120); only Jacobi.run's own awaits are sites` |
| `self.allreduce(...).get()` in the body | first execution | `Charm4PyError: Jacobi.run is a migratable coroutine: use 'await f' instead of f.get()` |
| an attribute dropped by `__getstate__` and read after a site | `+ck_fakemigrate`, 1 PE | the `AttributeError` from the user's line, after the rebuild |
| checkpoint restarted with a reordered await | `+restart` | the layout abort of 5.1 |
| a second `run` delivered while one is active | delivery | `Charm4PyError: Jacobi.run is already running on Jacobi(0, 1)` |

## 7. Open questions for Kale

| # | question | recommendation |
|---|---|---|
| Q1 | Re-run method: (a), (b), (c) or (d) of 3.3 | (d), the decoration-time guard transform |
| Q2 | Should every `async def` entry be a migratable coroutine, without `@coro(migratable=True)`? | not yet: require the decorator, reject a bare `async def` entry; revisit once the form has users |
| Q3 | Inbox and records in CkDyn (C++) or in the binding's pickled state | CkDyn: deliveries are matched without entering Python, the inbox moves as raw messages, and another binding (C, Julia) gets the same feature; a shared C++11 header with ckcoro only if the duplication grows past the ~300 lines estimated |
| Q4 | A chare at a pinning wait when `charm.checkpoint` runs: abort (as specified) or refuse the checkpoint with a list of the chares | abort in phase 1; a refusal needs a pre-pass over all chares that CkDyn does not have |
| Q5 | `async def main(args)` as a migratable coroutine of the main chare, so a program continues after `+restart` | yes, in phase 3, after the array form works; it needs `charm.start` to accept it |
| Q6 | Channels: pin forever, or make `recv` a site (reference number = port and sequence number) | pin in phase 1; a site in phase 2 if a ported example (jacobi2d with channels) needs it |
| Q7 | Helpers with their own sites (ckcoro's `CK_CALL`, chain depth > 0) | defer; `self.method` helpers without awaits cover the examples examined |
| Q8 | Explicit `site=` names instead of ordinals plus layout | no: ordinals plus the skeleton hash already survive edits between sites |
| Q9 | Drop the `migrateAndWait`/`sauerkraut` item (DESIGN.md 4.5 item 7) | drop it: this form needs no frame serialization and no third-party package |
| Q10 | Second invocation of an active migratable entry: error or queue | error, as ckcoro |
| Q11 | Propose the per-segment trace bracket (5.2) for ckcoro's `state::push` as well | yes, as a separate Charm++ PR when Kale lifts the no-PR rule |
| Q12 | A `+ck_fakemigrate` mode that really migrates at every site | add it to the binding's tests only (an env variable read by the test harness), not as a runtime flag |

## 8. Plan

1. **CkDyn** (Charm++ `ckdyn-dev`): everything in section 4; `doc/internals/ckdyn.md`
   gets a section "Coroutine records and the inbox". Test: a phase `coro` in
   `tests/charm++/ckdyn`, a C state machine over `CkDynCoroWait`/`Ready` on an array
   element: a migration while waiting (2 processes), a delivery buffered before the
   restart, `+ck_fakemigrate` at 1 PE, checkpoint and `+restart`.
2. **Transform** (charm4py `thin`, `charm4py/mcoro.py`): sites, rule checks, liveness
   pass, guards, layout hash. `tests/thin/mcoro_transform.py`, no runtime: each rule of
   3.2 and 3.5 rejected with its message; for a set of bodies, the function run with
   `_rs = k` executes no user statement before site k (a recording `self`).
3. **Driver** (charm4py): `@coro(migratable=True)`, `@inbox`, `self.inbox`,
   `charm.all`/`any`, element futures and their users (3.6), `contribute(ref=)`,
   pinning waits, the error for greenlet blocking calls, 5.3; registration sets
   `CKDYN_EP_CORO`/`CKDYN_EP_INBOX`, senders skip zero-copy for those entries. Python
   first; into `ckdyn_py.cpp` only if the coroutine message cost (DESIGN.md 7.4
   protocol) exceeds the greenlet form's 1.58 us.
4. **Tests** (`tests/thin/`):
   - `atsync_in_mcoro.py`, port of `atsync_in_coroutine.py`: phase 1 with `await
     self.AtSyncWait()`, every element changes PE and the `@when`-buffered message still
     arrives; phase 2 inverted: elements waiting at a site move, elements at a pinning
     wait stay and `migrate()` raises; `norecurse` unchanged.
   - `jacobi_mcoro.py`: 6.2 on a 4 x 4 grid, `lcrun -n 2 ... +pe 2 +balancer RotateLB`
     with `LB_PERIOD` 5, and `+pe 1 +ck_fakemigrate`; iteration count and temperature
     checksum equal the greenlet version's.
   - `mcoro_checkpoint.py`: checkpoint while waiting at sites, `+restart`, result equal
     to an uninterrupted run; a reordered await aborts with the layout message.
   - `mcoro_waits.py`: `charm.all`, `charm.any`, early deliveries, broadcast and
     point-to-point inbox sends, `charm.qd()`, element futures set from another PE,
     `contribute(ref=)`.
   - `gate4a.py` and `gate4c.sh` unchanged.
5. **Documentation**: new `docs/migratable-coroutines.rst` (form, rules, restart rule,
   the table of 3.6, checkpoint, `+ck_fakemigrate`); `docs/migrating.rst` "Suspended
   coroutines and migration" and "The AtSync pattern" point to it; DESIGN.md 4.5 item 7
   and the step 7 list refer to this file. The Charm++ user manual is unchanged.
6. **Ports**: migratable variants of examples/jacobi/jacobi2d.py and
   examples/wave2d/wave2d.py after steps 1-4 pass.

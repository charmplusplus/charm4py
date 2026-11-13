# SPEC-7-api: NodeGroup, entry-method attributes, message priorities, trace names

Step 7 items, written 2026-11-08 22:20 PST. Binding work on charm4py branch `thin` in
the main worktree; CkDyn already provides everything needed (DESIGN 3.5 attribute table,
3.6 `NodeGroupDyn`, `CkDynMsgBegin(..., prio_bits, ...)`, `CkDynMsgPriorityPtr`,
`CkDynCreateNodeGroup`, `CkDynSendNodeGroup`). A Charm++ change is expected only if a
test exposes a defect. No pull requests.

## 1. NodeGroup

```python
ng = NodeGroup(Class, args=[...])      # one branch per logical node (process)
ng[node].method(...)                     # node index 0..charm.numNodes()-1
ng.method(...)                           # broadcast to every branch
ng.method(..., ret=True)                 # gathered return values as for groups
```

- `Class` is a `Chare` subclass, as for `Group`. Inside a branch `self.thisIndex` is the
  node number and `self.thisProxy` the nodegroup proxy. `charm.numNodes()` and
  `charm.myNode()` are added (`CkDynNumNodes`/`CkDynMyNode`; add the two wrappers to
  `ckdyn.h` if absent, next to `CkDynMyPe`). With one PE per process (the current
  rule) node numbers equal PE numbers; the API must not assume it.
- Reductions: `self.contribute(...)` on a nodegroup branch contributes to the
  nodegroup's reduction, callback targets as for groups. `self.reduce`/`allreduce`
  work the same.
- `@entry(exclusive=True)` on a nodegroup method takes the branch's node lock
  (`CKDYN_EP_EXCLUSIVE`, `NodeGroupDyn::callEntry`). On other kinds it is an error at
  registration (`Charm4PyError`).
- Migration and checkpoint: nodegroups are packed and restored with `charm.checkpoint`
  like groups (DESIGN 3.10). A nodegroup entry may run on any PE of its node in a
  multi-PE process; that case does not exist yet (F14) and the binding keeps its
  `CmiMyNodeSize() == 1` check.
- `charm.nodegroups` dict parallel to `charm.groups`; appspector lists them.
- Reserved method names as for groups. `Chare(C, onPE=p)` is unchanged.

## 2. Entry-method attributes

`@entry(**attrs)` exists in `entry_method.py` and records the attributes; this step
makes registration honor them, mapped to `CKDYN_EP_*`:

| attribute | flag | note |
|---|---|---|
| `expedited=True` | EXPEDITED | every send to it gets `CK_MSG_EXPEDITED` |
| `inline=True` | INLINE | a local send runs the method at once; a coroutine entry with `inline` is a registration error, as in Charm++ ([threaded] and [inline] are exclusive) |
| `immediate=True` | IMMEDIATE | nodegroups only, else registration error; no effect on reconverse (DESIGN 3.5), accepted for source compatibility |
| `notrace=True` | NOTRACE | |
| `exclusive=True` | EXCLUSIVE | nodegroups only |
| `appwork=True` | APPWORK | |

Unknown attribute names raise `Charm4PyError` at registration. `@entry` composes with
`@coro` and `@when` in any order; registration reads `func._ck_entry_attrs` wherever it
is on the wrapper chain (set `__wrapped__` or copy the attribute in the other
decorators).

## 3. Message priorities

Per-send priority, a reserved keyword argument on proxy calls next to `ret`,
`awaitable` and `is_ray`:

```python
proxy.method(args, prio=p)        # p: int, lower value runs first (Charm++ integer priority)
proxy[i].method(args, prio=p)
```

- The binding reserves 32 priority bits (`CkDynMsgBegin(..., prio_bits=32, ...)`) and
  writes `p` as a signed 32-bit integer through `CkDynMsgPriorityPtr`; queueing is
  `CK_QUEUEING_IFIFO`, set through the send options CkDyn exposes (DESIGN 3.7, the
  `CkDynSendOpts` table around DESIGN line 625-660; if CkDyn has no field for
  queueing yet, add it in the same way as `prio`, as a C change on `ckdyn-dev`).
- Without `prio`, messages are sent as today (no priority bits; the default queue).
- Broadcasts, section multicasts and reduction callbacks accept `prio` too when CkDyn's
  send path carries it; otherwise `prio` on them raises `Charm4PyError` with a clear
  message. Say which in section 5.
- Coroutine resumes (future deposits) carry no user priority.

## 4. Trace names

Projections names (DESIGN 3.13) today: three `CHARE` lines per Python class (one per
kind) with the same name, and `charm4py.pool.Worker` collides with a user class named
`Worker`. Change: the registered trace name is the class's qualified name
(`module.Class`) unless the module is `__main__`, in which case it is `Class`; the
group and chare registrations of a class append `/group` and `/chare`, the array
registration keeps the plain name. Entry names stay `Class.method` with the same
class string. Check with the trace build (`charm/reconverse-darwin-arm8-charm4py-trace`,
`venv-trace`, program and decoder in `logs/step-7/trace/`) that the `.sts` file shows
the new names.

## 5. Tests and gate

- `tests/thin/nodegroup.py`: create, point-to-point and broadcast sends, `ret=True`
  gather, reduction to a future and to a main-chare method, `exclusive` entry with a
  check that two concurrent coroutine invocations on one branch do not interleave
  (the lock is taken for the synchronous part only; document what `exclusive` means
  for a coroutine entry), checkpoint and restart of a nodegroup, at 1, 2 and 4
  processes.
- `tests/thin/entry_attrs.py`: registration of every attribute, the three error cases,
  `expedited` and `inline` sends deliver, `inline` to a local element runs before the
  call returns (observable through a side effect), `notrace` in the `.sts` file (trace
  venv run, optional phase).
- `tests/thin/priorities.py`: a PE blocked in a coroutine receives 100 low-priority
  messages then one `prio=-1` message; the high-priority one must run first when the
  coroutine yields. Check at 1 and 2 processes.
- `docs/migrating.rst`: NodeGroup, `@entry`, `prio` move from "Removed or not yet
  available" to "New in 2.0"; close editor notes 4 and 5 at the end of the file (both
  answered: zero-copy arrays are read-only while pending; Python 3.11 or newer).
- Gate: `gate4a.py 1 2` 100 of 100 and `gate4c.sh 2`, plus the new tests in
  `test_config.json`.

## 6. Status

Status 2026-11-08: items 1-4 done on charm4py `thin` (85290de entry attributes,
623fbbe NodeGroup, c7cd56e priorities, 0be173a trace names) and Charm++ `ckdyn-dev`
89a5fbecf. The three tests pass at 1, 2 and 4 processes (`nodegroup.py` also with
`+restart` at each count); gate: `gate4a.py 1 2` 100 of 100, `gate4c.sh 2` 10 of 10 (logs/gate-4c/*-step7api.txt).

Different from the spec, or not stated by it:

- **Registration as a nodegroup.** Every class is now registered for NodeGroup as well
  as Group, Array and single chare, so `NodeGroup(C)` works for any registered class
  (one more `CHARE` line and entry set per class). A class with an
  `@entry(exclusive=True)` or `immediate` method is registered for NodeGroup only;
  registering it explicitly for another kind raises `Charm4PyError`. Without this
  rule the "error on other kinds" of section 2 would reject every such class, since
  the default registration includes all kinds. The checks run in `charm.register`,
  before anything is registered, so the error reaches the program as a Python
  exception and not as an abort inside the registration hook.
- **A Charm++ change was needed** (89a5fbecf): `CkDynGetReductionNumber` aborted for
  a nodegroup branch, which `allreduce` needs. `NodeGroupDyn::getRedNo()` exposes
  the protected `reductionInfo.redNo`; array maps take the group path. The
  `tests/charm++/ckdyn` nodegroup constructor checks it. `CkDynMyNode` and
  `CkDynNumNodes` already existed.
- **Nodegroup ids are a separate id space** from group ids (`_numNodeGroups` and
  `_numGroups` in ck.C), so branches live in `charm.nodegroups`, not `charm.groups`,
  and appspector keys them as `('n', gid)`; in the ObjLoads reply a branch is
  reported as a group branch with its nodegroup id.
- **exclusive on a coroutine entry**: CkDyn holds the node lock for the delivery
  call, which returns at the coroutine's first suspension; a resume runs outside the
  lock. So each synchronous part runs alone, and a second invocation can start while
  the first is suspended. `nodegroup.py` checks the second part (the first is trivial
  with one PE per process); the user docs say so.
- **Priorities need no Charm++ change.** `CkDynMsgSetQueueing` exists; ckdyn.h has
  no queueing constants, so the binding uses the value 4 (`CK_QUEUEING_IFIFO`) as
  `tests/charm++/ckdyn` does. Point-to-point sends and broadcasts to groups,
  nodegroups and arrays carry `prio`; a section send with `prio` raises
  `Charm4PyError`; contributions have no `prio` parameter, so reduction callbacks
  carry no priority. `prio=None` sends without one; a non-int raises `TypeError`, a
  value outside 32 bits `OverflowError`.
- **Priorities across processes are best-effort.** Decision 2026-11-08 22:50 PST
  (Kale): a priority is a hint and guarantees nothing, as in Charm++; no reconverse
  change is asked for. The observation stays for the record. Within a PE the
  prio=-1 message runs before 100 queued plain ones. When the 101 messages come from
  another process while the receiver is busy, the prio=-1 message ran at positions
  48 to 99 of 101 in six runs (24 with `+old-scheduler`): the communication backend
  hands received messages to the scheduler in batches, one per progress call, and
  the priority queue runs between batches, so it never holds all of them. The test
  checks delivery and prints the position. Fixing it belongs in reconverse (drain
  the backend before taking from the priority queue, or bound the drain); not
  attempted here.
- **Trace names**: the nodegroup registration appends `/nodegroup`; the main-chare
  registration keeps the plain name, as arrays do. appspector's "SumDetail sts"
  reply uses the same class strings. The migration-constructor entry that CkDyn
  registers carries the registered chare name (`Worker/group`). Verified in
  `logs/step-7/trace/names` with the trace build and `venv-trace`: the `.sts` file
  lists `charm4py.pool.Worker/group`, `charm4py.pool.Worker`, `.../chare`,
  `.../nodegroup`, `Worker`, `Worker/group`, `Worker/chare`, `Worker/nodegroup`,
  `NX/nodegroup`, entries `Worker.poke`, `charm4py.pool.Worker.runTask`; the log has
  4 events for `Worker.poke` and none for the `@entry(notrace=True)` method
  `Worker.quiet`, called as often. `logs/step-7/trace/decode.py` now accepts the new
  name of `CharmRemote`.
- `_ckdyn.ep_flags(ep)` was added so that tests can read the registered flags.

# Step 4c specification: migration, load balancing and checkpoint

Written 2026-11-08 with the implementation (branch `thin`), against `ckdyn-dev`
c81fbeecc. It extends SPEC-4a.md and SPEC-4b.md. DESIGN.md 3.10, 3.12, 4.5 (items
1-6) and 5.4 items 10, 13 and 14 are the design; review findings F9 and F10
(REVIEW-2026-11-07.md) were implemented in the 4a follow-ups and are verified here.

## 1. What migration does in the binding

- CkDyn's `pupObj` (ckdyn.C) pups the chare index, the section cookies and then the
  binding's bytes through the hooks `pack_size`, `pack` and `unpack`
  (doc/internals/ckdyn.md "Pup"). The reason is `CKDYN_PUP_MIGRATE` or
  `CKDYN_PUP_CHECKPOINT`; the sizing pass ORs in `CKDYN_PUP_SIZING`.
- `h_pack_size` calls `charm._pack(obj, reason)` with the full reason on every
  pass and keeps the returned bytes only until `h_pack` copies them into the
  pupper. `_pack` pickles the object (protocol `charm.options.pickle_protocol`,
  default 5) with `_ckdyn_target` removed from its `__dict__`. The bytes of a
  sizing pass are kept on the object (`_ckdyn_packed`, with the reason) and
  returned by the packing pass of the same reason, so a migration pickles once.
  A sizing pass can also run alone: `CkMigratable::AtSync` sizes every element
  with a migration pupper under `+MetaLB`. Therefore a sizing pass never refuses
  and has no other effect, and the packing pass checks again whether the object
  may be packed (section 3).
- The element leaves `charm.arrays` in `_destruct`, when the runtime deletes it
  after it has migrated, not in `_pack` (step 4a removed it in `_pack`, which a
  sizing pass alone would also have done).
- `_unpack` unpickles, sets `_ckdyn_target` to the new target, registers the
  object in `charm.arrays` (or `groups`, `chares`) and, for a migration, runs
  `migrated()` through its `EntryMethod` (profiled like any entry). `thisIndex`,
  `thisProxy`, numpy attributes and `__getstate__`/`__setstate__` behave as in
  1.x: they are pickled state. The main chare and groups use the same path for
  checkpoint and restart.
- An attribute that cannot be pickled aborts the program with
  `Charm4PyError: cannot pickle <Class><index> for migration: <error>` (step 4a
  aborted with the bare pickle error). A class that needs such an attribute
  drops it in `__getstate__` and recreates it in `__setstate__`
  (tests/thin/atsync_in_coroutine.py does this with a `threading.Lock`).
- `self.migrate(pe)` calls `CkMigratable::ckMigrate`, which packs and deletes
  the element before it returns, as in 1.x and in C++. Code after the call in the
  same entry runs on the Python object that has left; its changes are lost, and a
  contribution from it aborts (`CkDyn: contribute from an object that is not
  local`). This includes the reduction of an `awaitable=True` broadcast to an
  entry that migrates its element. Call `migrate` last, and report completion
  from `migrated()`.

## 2. AtSync and resumeFromSync

- `Chare.AtSync()` is `_ckdyn.at_sync(self)` → `CkDynAtSync` →
  `ArrayElement::AtSync()`. Arrays created with `useAtSync=True` set
  `usesAtSync` through `CKDYN_MSG_USESATSYNC`.
- `ArrayElemDyn::ResumeFromSync` calls the `resume_from_sync` hook, and
  `charm._resume_from_sync(obj)` sends a message to the element's
  `resumeFromSync` entry; it never calls it. With no load balancer
  `CkMigratable::AtSync` calls `ResumeFromSync` inside the `AtSync` call, so a
  synchronous call would nest one level per iteration for programs whose
  `resumeFromSync` calls `AtSync` again (tests/migration/test_migrate.py);
  with a balancer, user code would run inside the barrier's resume loop (F10).
  `atsync_in_coroutine.py norecurse` runs 3000 such iterations and checks that
  the Python stack depth does not grow.
- Every element with `useAtSync` on a PE takes part in that PE's barrier, as in
  Charm++: a program with two AtSync arrays must call `AtSync` on both.

## 3. Migratability while a coroutine is suspended

- `threads.pauseThread` increments `obj._nsuspended` before it switches to the
  main greenlet and decrements it after it is resumed. On 0 → 1 it calls
  `_ckdyn.set_migratable(obj, False)`, on 1 → 0 `set_migratable(obj, True)`,
  both only if `obj.migratable` is true (`setMigratable(False)` by the user is
  kept). A coroutine that runs to completion never toggles anything, so a
  coroutine that calls `AtSync()` and returns (examples/particle/particle.py) is
  migrated by the balancer (F9).
- `obj._numthreads` keeps its 1.x meaning (coroutines started and not finished,
  read by two tests); it plays no part in migration.
- `migrate(pe)` with `_nsuspended > 0` raises `Charm4PyError("Cannot migrate
  <Class><index>: it has N suspended coroutine(s)")` before anything is sent.
- A packing pass with reason MIGRATE and `_nsuspended > 0` returns `None`, and
  CkDyn aborts with "the object cannot be packed in its current state". This can
  only happen in the window DESIGN.md 4.5 item 3 describes.
- CHECKPOINT always packs; after `+restart` `_numthreads` and `_nsuspended` are 0.

## 4. Section members

Nothing in the binding: CkDyn pups each element's section cookies with the
element, and CkMulticast rebuilds its tree after members move (internals chapter
"Sections: Migration and checkpoint"). tests/thin/sections_migrate.py splits an
array into two sections, combines them, and runs five rounds of section
multicasts and section reductions with every element moved by RotateLB between
rounds.

## 5. When conditions and wait queues

Messages buffered behind a `@when` condition and the condition objects
(`_cond_next` list) are pickled with the element (wait.py `__getstate__`), as in
1.x. atsync_in_coroutine.py phase 1 buffers one message per element before each
balancing period and checks that it is delivered after the period, on the new PE,
by the auto-flush that follows `resumeFromSync`. No test sets
`auto_flush_wait_queues = False`; with it off, the buffered messages wait for an
explicit `__flush_wait_queues__()`, as in 1.x.

## 6. Defects found outside the binding

- Runtime (also in plain C++ on `recharm`): an element moved by
  `migrateMe`/`ckMigrate` that then calls `AtSync` hangs the next load-balancing
  step on reconverse at two processes: the PE that receives the moved elements
  resumes only some of its elements and the other PE resumes none. On ckdyn-dev
  c81fbeecc the run hangs; on recharm 08364f246 (build of 2026-10-23) it aborts
  with `Assertion numLevels > 0 && awaitingLB[level] failed` (TreeLB.C:484).
  Reproduction: logs/gate-4c/cxx-repro/mig.C (`lcrun -n 2 ./mig 1 +pe 2
  +balancer RotateLB`; argument 0, no manual migration, passes). Not a CkDyn
  defect; atsync_in_coroutine.py avoids the pattern.
  Update 2026-11-08 14:15 PST: fixed on charm branch `treelb-unordered-arrival`
  (7192d2281 on recharm 16f27646b), issue charm #4195. Cause: TreeLB counted every
  arrival as a migration it ordered, so a `migrateMe` arrival corrupted its per-step
  count. Fix: `CkLocMgr::emigrate` marks arrivals the balancer did not order and
  balancers ignore them. `mig.C` passes 8 of 8 on that branch. ckdyn-dev picks it up at
  the next rebase onto recharm after the merge. A `migrateMe` while an LB step is in
  progress on the destination PE remains unsafe; between steps it is fine.
- `+MetaLB` with examples/particle/particle.py aborts in `CkMigratable::AtSync`
  ("Contributing to iter 0 < current iter 1"); tests/migration/test_migrate.py
  passes with `+MetaLB`. Not investigated; `+MetaLB` is not in any gate.

## 7. Received numpy arrays kept alive

A received array is a view of the runtime message, which the binding retains
(`CkDynMsgRetain`) for as long as the array lives. tests/thin/views_retained.py:
PE 1 sends N messages of 64 KB float64 arrays (random values, so that macOS does
not compress the pages) to PE 0 in windows of 500; PE 0 keeps them. Two
processes, this Mac, 2026-11-08:

| mode (N) | wall time | receiver RSS before → after |
|---|---|---|
| drop (10000) | 0.79 s | 88 → 90 MB |
| copy (10000), payload 625 MB | 1.06 s | 87 → 716 MB |
| keep views (10000), payload 625 MB | 0.83 s | 87 → 872 MB |
| keep views (30000), payload 1875 MB | 3.56 s | 88 → 1244 MB (peak 1797 MB) |

Every run completes; nothing stalls, so the receive path does not hold an LCI or
reconverse receive buffer: the message is an ordinary runtime allocation once it
is delivered. A kept view costs its whole message, about 15 KB (23 %) more per
64 KB array than a copy here (872 against 716 MB) (message header and allocation rounding). The
30000 run's RSS is below the payload because macOS compresses and pages out under
memory pressure. No change is proposed; a program that keeps many small received
arrays can set `charm.options.copy_received_arrays = True` or copy them itself.

## 8. Other changes

- `Chare.__init__(obj)` is called directly before the user constructor (was
  `super(em.C, obj).__init__()`, which for a chare class that inherits its
  constructor from another chare class ran the parent's constructor without
  arguments first). 1.x has the same defect.

## 9. Tests

| test | what |
|---|---|
| tests/migration/test_migrate.py | AtSync every 20 iterations, `resumeFromSync` calls `start()` → `AtSync`; numpy state checked |
| tests/migration/test_nonmigratables.py | `setMigratable(False)` elements stay; `migrated()` and `hasMigrated` |
| tests/migration/chare_migration.py | `migrate(pe)` through a proxy call; `migrated()` on the new PE |
| examples/particle/particle.py | coroutine calls `AtSync` and returns; channels re-established across moves |
| tests/thin/atsync_in_coroutine.py | F9, F10 (`norecurse`), state after migration, `@when` queue, suspended elements skipped and `migrate()` raising, subclass constructor |
| tests/thin/sections_migrate.py | section and combined-section multicasts and reductions across RotateLB moves |
| tests/thin/views_retained.py | section 7 |
| tests/thin/checkpoint_restart.py | checkpoint and `+restart` through the same `_pack`/`_unpack` |

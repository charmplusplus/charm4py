# SPEC-7-device: persistent device buffers, posted receives, IPC path

Step 7 item, written 2026-11-08 21:35 PST after step 3 (SPEC-3.md). Kale rates this
item as important for GPU performance. Layer 2 work on a Charm++ branch `ckdyn-dev7`
(worktree `charm-dev7`), binding work on charm4py branch `thin-dev7` (worktree
`charm4py-dev7`); both merge into `ckdyn-dev` and `thin` when done. No pull requests.

## 1. Scope

In: (a) the persistent direct device API, Charm++'s `CkDevicePersistent` (get and put
between registered device buffers, no entry method, no per-transfer registration);
(b) a Python surface for receiver-posted host and device buffers, over
`CkDynPostBuffer` and `CkDynPostDeviceBuffer` from step 3; (c) the `+gpushm` IPC path
for DEVICE slots and persistent transfers between processes on one node, never run so
far; (d) measurements that say when each path pays.

Out: HIP runs (code stays portable through `hapi_portable.h`), wheels with device
support, changes to the DEVICE slot semantics of SPEC-3.

## 2. Persistent device buffers

Charm++ side (read `src/ck-core/ckrdmadevice.h/.C`, `CkDevicePersistent`, and the users
under `examples/charm++/cuda/gpudirect/` and `tests/charm++/cuda/` before designing):
a `CkDevicePersistent` wraps a device pointer and count, is registered with `open()`
and released with `close()`, pups as metadata so it can be sent to another chare, and
`dst.get(src)` or `src.put(dst)` moves the bytes, invoking the buffer's `CkCallback`
on completion. Transfer mode is memcpy on one PE, IPC in one process group with
`+gpushm`, RDMA otherwise.

C API additions to `ckdyn.h` (no change to `CkDynHooks`, `CKDYN_API_VERSION` stays 2):

```c
typedef struct CkDynDevPersist CkDynDevPersist;      /* opaque, PE-local */
CkDynDevPersist *CkDynDevPersistOpen(const CkDynDevDesc *d, const CkDynCallback *done);
void   CkDynDevPersistClose(CkDynDevPersist *p);
/* portable handle: fixed-width bytes a binding may send inside a message */
size_t CkDynDevPersistHandleSize(void);
void   CkDynDevPersistHandle(const CkDynDevPersist *p, void *out);
/* dst must be local (opened here); src is a handle received from anywhere */
void   CkDynDevPersistGet(CkDynDevPersist *dst, const void *src_handle, uint32_t refnum);
void   CkDynDevPersistPut(const void *dst_handle, CkDynDevPersist *src, uint32_t refnum);
```

`done` is invoked once per completed get or put on the PE that issued it, with an
empty `CkDynMsg` whose header `refnum` is the `refnum` of the call, so the binding can
match completions to futures (the slot mechanism of SPEC-4b for section callbacks is
the pattern). If `CkDevicePersistent` fires its callback only on one side, say which
in `ckdyn.md`; the binding needs the issuing side.

Python (`charm4py/charm.py`, `_ckdyn`):

```python
p = charm.DevicePersistent(cupy_or_torch_array)    # opens; array must stay alive
h = p.handle                                        # picklable, sendable as an argument
fut = p.get(h_remote)                               # Future, resolved with None when the bytes have arrived
fut = p.put(h_remote)
p.close()                                           # also at garbage collection, with a warning
```

One outstanding transfer per local buffer; a second before the first completes raises
`Charm4PyError`. `DevicePersistent.handle` is a small immutable object with
`__reduce__`, so `proxy.method(p.handle)` works and the receiver gets a handle it can
pass to its own buffer's `get`. The array the buffer wraps must stay contiguous and
alive until `close()`.

## 3. Posted receives

```python
self.post_buffer(self.method, slot, array)   # array: numpy (host) or __cuda_array_interface__ (device)
```

Registers `array` as the destination of slot `slot` (0-based positional argument index
of the entry method, counting only arguments; the binding maps it to the CkDyn slot
index) for the next message to this element invoking `method`. The method is then
called with that same Python object as the argument, not a fresh view, so identity is
preserved and no copy is made for device buffers. Host posts cost one copy on the
receiving PE (SPEC-3 section 7, deviation 1), which still saves the message-sized
allocation and the second copy out of the view. Lengths must match, else the program
aborts with both lengths, as the C API does. One outstanding post per (method, slot);
posts are dropped on migration (the C API drops them) and `post_buffer` raises if
called on a chare that is not local. Group branches may post; nodegroups may not
(C API).

## 4. The `+gpushm` IPC path

Run the step 3 `device` phases and `tests/thin/device.py`, plus the persistent test of
section 5, with `+gpushm` at two processes on one Delta node, both GPUs. Record which
transfer mode ran (the device code prints registration lines only on the RDMA path; add
a one-line `+gpushm` confirmation print in CkDyn's device path if nothing identifies
IPC) and the one-way times against the RDMA path on the same node.

## 5. Tests

- `tests/charm++/ckdyn/`: a `devpersist` phase (CUDA/HIP guard): two elements on
  different PEs each open a 16 MB buffer, exchange handles through an entry, element 1
  gets from element 0, checks the pattern, then puts back, element 0 checks; same PE
  case; close and reopen.
- `tests/thin/device_persistent.py` (skipped without device support or cupy): the same
  in Python with cupy arrays, plus sending a handle inside a list (pickled) and the
  "second transfer while one is outstanding" error.
- `tests/thin/posted_receive.py`: host post with numpy at 1 and 2 processes (runs on
  the Mac), device post with cupy (Delta), identity check (`received is posted`),
  length mismatch abort, post then migrate (post dropped, next message allocates).
- Delta jobs through `tools/delta/gpu-gate.sbatch` with FILTER, one node two GPUs and,
  for persistent get over the fabric, two nodes.

## 6. Measurements (Delta, record in section 7)

16 MB device to device, one way, medians of 20 after warm-up: DEVICE slot message
(SPEC-3 path), persistent `get`, both for one node two processes with and without
`+gpushm`, and two nodes. Also persistent `get` of 1 MB, 4 MB, 64 MB on two nodes.
The question to answer: how much the persistent path saves per transfer against the
message path (registration and descriptor cost), and whether `+gpushm` beats RDMA on
one node.

## 7. Status

Status 2026-11-08 23:10 PST: done. Sections 2-6 built, tested on the Mac (host side) and
on NCSA Delta (A100, CUDA 13.2), including `+gpushm` and two nodes. Then, at Kale's
decision (2026-11-08 22:35), the three-mode implementation moved into Charm++'s
`CkDevicePersistent` with CkDyn wrapping it (step 4b), and the one-node job was rerun.

Commits. Charm++ `ckdyn-dev7`: 669707835 (CkDyn persistent buffers, argument posts,
unpost, `devpersist` phases), bbcc2c9ed (step 4b: implementation in `CkDevicePersistent`,
CkDyn wrappers, manual). charm4py `thin-dev7`: beef19d (binding), 75605e5
(`device_persistent.py`), 67c9813, b4b1263, c5a29ee (Delta scripts), f0d5890 and this
commit (docs). Mechanism: `doc/internals/ckdyn.md`, "Persistent device buffers" and
"Posts". Logs: `logs/step-7/device/` (`mac/`, `devp-<job>/`, `devp-<job>.out`).

**Built.**

- C API of section 2, plus `CkDynDevPersistMode` (the mode a transfer would use). The
  handle is a fixed 512-byte record holding the pupped `CkDevicePersistent` (pointer,
  length, PE, process, host, the IPC handle of the allocation and the buffer's offset in
  it, and the `CmiNcpyBuffer` with its registration) with an ignore callback.
- `done` semantics: once per get or put on the issuing PE; for a `FUTURE` callback the
  call's refnum is the future id, for any other kind the callback's refnum. The binding
  opens every buffer with a `FUTURE` callback to its own PE and passes each transfer's
  future id as the refnum, so no slot table is needed.
- `CkDevicePersistent` (step 4b): public constructors, `open`/`close`/`set_msg`/`get`/
  `put` and the manual's callback semantics (both buffers' callbacks sent from the
  issuing PE with their `cb_msg`) kept; added `mode()`. Defects fixed: it aborted between
  hosts; it opened the IPC handle of an interior pointer as if it were the allocation
  base (wrong for cupy, torch and pool allocations); it kept the IPC mapping in the
  peer's copy, so every copy opened the handle again and `close()` on a copy unmapped
  it. Also: the stream was uninitialized in the constructors without one.
- Binding: `Chare.post_buffer`, `charm.DevicePersistent`, `charm.DevicePersistentHandle`,
  `_ckdyn.DevPersist`, `_ckdyn.post_buffer`, `_ckdyn.posted_count` (tests).
- Tests: `ckdyntest` phases `devpersist/remote` and `devpersist/samepe`;
  `tests/thin/posted_receive.py`, `tests/thin/device_persistent.py`;
  `tools/delta/gpu-devpersist.sbatch` (also runs Charm++'s
  `examples/charm++/cuda/gpudirect/persistent`), `tools/delta/bench_devpersist.py`,
  `tools/delta/build-incr.sbatch`.

**Deviations, each kept.**

1. The transfer code is new, not a wrapper of the old `CkDevicePersistent` (which aborts
   between hosts and mishandles interior pointers and IPC mappings); since step 4b it is
   the new `CkDevicePersistent`, so C++ programs have it too.
2. The IPC mode is used only with `+gpushm`, as section 2 states, although persistent
   IPC does not use the `+gpushm` communication buffer; without it, two processes on one
   host use RDMA. For C++ programs this changes the old `CkDevicePersistent`, which
   used IPC between processes on one host without the flag.
3. An RDMA put is a get issued by the destination's PE (two Converse handlers in
   ckrdmadevice.C): a true RDMA put completes on the source before the bytes are
   visible at the destination (reconverse `CommRputLocalHandler`), so its callbacks
   could not mean "the bytes have arrived". Cost: one small message each way, and the
   put measured no slower than the get (tables below).
4. Two C additions besides section 2: `CKDYN_POST_ARG` (a post counts arguments: a
   `CKDYN_MSG_BLOCK` message's leading `FUTURE` slot is skipped, which the binding cannot
   know at post time) and `CkDynUnpostBuffer` (a post that a `BUFFER` slot did not
   consume is removed by the binding, which then copies the value itself).
5. A posted argument that arrives as a copy (host array below the threshold, any array
   in a broadcast) is copied into the posted array with `obj[...] = value`, so identity
   holds on every path; the length check then compares `nbytes`.
6. `DevicePersistent.get`/`put` create their future with `charm.createFuture`, so they
   are called from a coroutine, as every Future-returning call in 1.x.

**Compile on Delta.** Everything under `CMK_CUDA || CMK_HIP` compiled without a change or
a new warning, before and after step 4b (build jobs 23677297, 23677399, both on
`cpu-interactive`: the auto-mode classifier refused a `make -j16` on the login node, and
the GPU interactive queues had a two-hour wait at the time). No run-time defect was
found; no Job A iteration was needed.

**Results, one node** (gpuA100x4-interactive, A100-SXM4-40GB, `srun --mpi=pmix`).
Shapes: `n1` one process; `n2` two processes, one GPU per task; `n2vis` two processes
with both GPUs visible to both (HAPI's block mapping gives process k device k);
`n2visshm` the same with `+gpushm`. Job 23677329 at 669707835, job 23677400 at bbcc2c9ed
(step 4b); results identical.

| test | n1 | n2 | n2vis | n2visshm |
|---|---|---|---|---|
| ckdyntest `ncpy/*`, `device/*`, `hapi` | PASS | PASS | PASS | PASS |
| ckdyntest `devpersist/remote` (mode) | PASS (copy) | PASS (RDMA) | PASS (RDMA) | PASS (IPC) |
| ckdyntest `devpersist/samepe` | PASS (copy) | PASS (copy) | PASS (copy) | PASS (copy) |
| `examples/charm++/cuda/gpudirect/persistent` (4b only) | | PASS (RDMA) | PASS (RDMA) | PASS (IPC) |
| `device_persistent.py` | PASS (copy) | PASS (rdma) | PASS (rdma) | PASS (ipc) |
| `posted_receive.py` (host and device phases) | PASS | PASS | PASS | PASS |
| `device.py` (DEVICE slot mode) | PASS (copy) | PASS (RDMA) | PASS (RDMA) | PASS (IPC) |
| `posted_receive.py -mismatch` | aborts with both lengths | | | |

Two nodes (job 23677376, gpua001 and gpua009, one A100 each, at 669707835): ckdyntest
all phases PASS (`devpersist/remote` RDMA), `device_persistent.py`, `posted_receive.py`,
`device.py` PASS.

**`+gpushm`.** The transfer mode is printed once per process: `CkDyn> DEVICE slots: ...`
and `CkDevicePersistent> get/put: ...`. With `+gpushm` and both GPUs visible, DEVICE
slots and persistent transfers between the two processes take the IPC path (first runs
of the IPC path for DEVICE slots). With one GPU per task, `+gpushm` aborts in `hapiInit`
("Fatal hapi Error invalid argument", hapi_impl.cpp:1739, `ipcHandleOpen`): HAPI opens the
other process's communication buffer, whose GPU that process cannot see. So `+gpushm`
needs `srun --gpus=2 --gpu-bind=none` (both GPUs visible), not `--gpus-per-task=1`.

**Measurements** (16 MB unless stated; one-way, median of 20 after 3 warm-up). DEVICE
slot: `bench_zc_delta.py device`, round trip / 2 between main and element 1. Persistent:
`bench_devpersist.py`, element 1 times the call to the future's value (done callback
included), each element's cupy array on the device HAPI chose (printed in the logs).

One node (job 23677400, gpua086):

| path | n2 (RDMA, one GPU per task) | n2vis (RDMA) | n2visshm (IPC) |
|---|---|---|---|
| DEVICE slot (ms) | 0.820 | 0.821 | 0.267 |
| persistent get (ms) | 0.848 | 0.849 | 0.199 |
| persistent put (ms) | 0.748 | 0.750 | 0.200 |

Two nodes (job 23677376, Slingshot):

| size | persistent get (ms) | persistent put (ms) | DEVICE slot (ms) |
|---|---|---|---|
| 1 MB | 0.069 | 0.084 | |
| 4 MB | 0.244 | 0.237 | |
| 16 MB | 0.848 | 0.857 | 0.895 |
| 64 MB | 4.085 | 4.147 | |

Answers. Between nodes, and between processes on one node over RDMA, the persistent path
saves little against a DEVICE slot message at 16 MB (0.85 against 0.90 ms between nodes;
on one node the get is no faster, the put 0.07 ms faster): the transfer itself dominates,
and the message path's registration of the pool arena and per-message descriptors cost
less than 0.1 ms. On one node, IPC (`+gpushm`) beats RDMA by 3 to 4 times (persistent
0.20 against 0.75-0.85 ms; DEVICE slot 0.27 against 0.82 ms), and there the persistent
path saves 0.07 ms (25%) against the DEVICE slot, whose IPC path stages through the
`+gpushm` communication buffer. Two nodes reach 16 MB / 0.85 ms = 19.7 GB/s, 64 MB 16.4 GB/s.

Caveat on measurement. The first one-node job measured persistent IPC at 0.042 ms because
both processes' cupy arrays were on device 0 (cupy's default device is 0 even when HAPI
set another); `bench_devpersist.py` now selects the device HAPI chose. Python programs
with several visible GPUs per process must do the same (`cupy.cuda.Device(
cupy.cuda.runtime.getDevice()).use()`). The DEVICE slot one-node RDMA numbers differ
between nodes (1.23 ms on gpua011 in job 23677329, 0.82 ms on gpua086 in job 23677400);
the tables compare paths within one job.

**Open.** HAPI's `+gpushm` with one GPU per task (above). cupy's default device in
multi-GPU processes (a note for docs/gpus.rst, or the binding could select HAPI's device
at start-up). The pin in `charm_version.txt` must move to the merged ckdyn-dev commit (the
binding needs `CkDynUnpostBuffer`). Two-node runs were not repeated after step 4b (the
RDMA paths ran in the one-node `n2`/`n2vis` shapes of job 23677400).

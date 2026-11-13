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

Status 2026-11-09 00:05 PST: done. Sections 2-6 built, tested on the Mac (host side) and
on NCSA Delta (A100, CUDA 13.2), one node with and without `+gpushm` and two nodes. Then,
at Kale's decisions of 2026-11-08: the implementation moved into Charm++'s
`CkDevicePersistent` with CkDyn wrapping it (step 4b); `CkDevicePersistent` uses IPC
between processes on one host without `+gpushm`, as it did before, whenever the peer's
IPC handle opens; and the binding makes HAPI's device the current CUDA device of each PE
(`charm.myDevice()`).

Commits. Charm++ `ckdyn-dev7`: 669707835 (CkDyn persistent buffers, argument posts,
unpost, `devpersist` phases), bbcc2c9ed (step 4b), b0412d559 (IPC without `+gpushm`),
de12c56a4 (drops an unverified claim of bbcc2c9ed, below), 8837b899e
(`CkDynHapiMyDevice`, `CkDynHapiUseMyDevice`). charm4py `thin-dev7`: beef19d (binding),
75605e5 (`device_persistent.py`), 4ef90ce (`charm.myDevice()`), Delta scripts 67c9813,
b4b1263, c5a29ee and three later, docs f0d5890, acde733 and this commit. Mechanism:
`doc/internals/ckdyn.md`, "Persistent device buffers", "The device of a PE" and "Posts".
Logs: `logs/step-7/device/` (`mac/`, `devp-<job>/`, `devp-<job>.out`).

**Built.**

- C API of section 2, plus `CkDynDevPersistMode` (the mode a transfer uses) and
  `CkDynHapiMyDevice`/`CkDynHapiUseMyDevice`. The handle is a fixed 512-byte record
  holding the pupped `CkDevicePersistent` (pointer, length, PE, process, host, the IPC
  handle of the allocation and the buffer's offset in it, and the `CmiNcpyBuffer` with
  its registration) with an ignore callback.
- `done` semantics: once per get or put on the issuing PE; for a `FUTURE` callback the
  call's refnum is the future id, for any other kind the callback's refnum. The binding
  opens every buffer with a `FUTURE` callback to its own PE and passes each transfer's
  future id as the refnum, so no slot table is needed.
- `CkDevicePersistent`: public constructors, `open`/`close`/`set_msg`/`get`/`put` and the
  manual's callback semantics (both buffers' callbacks sent from the issuing PE with
  their `cb_msg`) kept; `mode()` added. Mode: a device copy within a process; CUDA/HIP
  IPC between processes on one host when the peer's IPC handle opens in this process
  (with or without `+gpushm`); RDMA on the memory `open()` registered otherwise. Defects
  fixed: it aborted between hosts; it opened the IPC handle of an interior pointer as if
  it were the allocation base (wrong for cupy, torch and pool allocations); it kept the
  IPC mapping in the peer's copy, so every copy opened the handle again and `close()` on
  a copy unmapped it; the stream was uninitialized in the constructors without one.
  bbcc2c9ed's message (and a comment and ckdyn.md, corrected in de12c56a4) said CUDA
  refuses a second open of a handle in one process; that was not observed.
- Binding: `Chare.post_buffer`, `charm.DevicePersistent`, `charm.DevicePersistentHandle`,
  `charm.myDevice()`, `_ckdyn.DevPersist`, `_ckdyn.post_buffer`, `_ckdyn.posted_count`.
  Before the first Python code on a PE the binding calls `CkDynHapiUseMyDevice`, so cupy,
  torch and numba allocate on the PE's GPU by default.
- Tests: `ckdyntest` phases `devpersist/remote` and `devpersist/samepe`;
  `tests/thin/posted_receive.py`, `tests/thin/device_persistent.py` (which also asserts
  cupy's current device equals `charm.myDevice()` on every PE); Delta scripts
  `gpu-devpersist.sbatch` (also runs Charm++'s `examples/charm++/cuda/gpudirect/persistent`),
  `bench_devpersist.py` (keeps its explicit device selection as a check),
  `build-incr.sbatch`.

**Deviations, each kept.**

1. The transfer code is new, not a wrapper of the old `CkDevicePersistent` (which aborts
   between hosts and mishandles interior pointers and IPC mappings); since step 4b it is
   the new `CkDevicePersistent`, so C++ programs have it too.
2. Section 2 tied IPC to `+gpushm`. At Kale's decision the persistent API does not look
   at `+gpushm`: IPC whenever the peer's handle opens, RDMA when it does not (one GPU per
   task). `+gpushm` keeps its meaning for DEVICE-slot messages only.
3. An RDMA put is a get issued by the destination's PE (two Converse handlers in
   ckrdmadevice.C): a true RDMA put completes on the source before the bytes are
   visible at the destination (reconverse `CommRputLocalHandler`), so its callbacks
   could not mean "the bytes have arrived". Cost: one small message each way; the put
   measured no slower than the get (tables below).
4. Two C additions besides section 2 for posts: `CKDYN_POST_ARG` (a post counts
   arguments: a `CKDYN_MSG_BLOCK` message's leading `FUTURE` slot is skipped, which the
   binding cannot know at post time) and `CkDynUnpostBuffer` (a post that a `BUFFER` slot
   did not consume is removed by the binding, which then copies the value itself).
5. A posted argument that arrives as a copy (host array below the threshold, any array
   in a broadcast) is copied into the posted array with `obj[...] = value`, so identity
   holds on every path; the length check then compares `nbytes`.
6. `DevicePersistent.get`/`put` create their future with `charm.createFuture`, so they
   are called from a coroutine, as every Future-returning call in 1.x.

**Compile on Delta.** Everything under `CMK_CUDA || CMK_HIP` compiled without a change or
a new warning at every commit (build jobs 23677297, 23677399, 23677525 on
`cpu-interactive`: the auto-mode classifier refused a `make -j16` on the login node).
No run-time defect was found in the CkDyn or binding code.

**Test-harness defects found and fixed** (scripts only): `ckdyntest`'s Makefile relinks
only when an object changes, so jobs 23677400, 23677526 and 23677847 ran the `ckdyntest`
linked in job 23677329 (before step 4b); the script now cleans first, and job 23678146
ran the current one. `srun --gpus=2 --gpu-bind=none` gave both tasks the same single GPU
on gpua040 (job 23677526: HAPI "1 device(s) per host", both PEs on bus 85:00.0, so its
IPC numbers were copies within one GPU); the `n2vis` shapes now add
`CUDA_VISIBLE_DEVICES=0,1` inside the step and the script prints what each task sees and
HAPI's configuration line.

**Results, one node** (gpuA100x4-interactive, A100-SXM4-40GB, `srun --mpi=pmix`).
Shapes: `n1` one process; `n2` two processes, one GPU per task; `n2vis` two processes,
both GPUs visible to both (HAPI gives process k device k; checked: PE 1 on device 1,
another bus); `n2visshm` the same with `+gpushm`. Final rule (b0412d559, 8837b899e,
4ef90ce): job 23678146 (tests) and 23677847 (measurements); `n1` from job 23677329 (before step 4b; not rerun).

| test | n1 | n2 | n2vis | n2visshm |
|---|---|---|---|---|
| ckdyntest `ncpy/*`, `device/*`, `hapi` | PASS | PASS | PASS | PASS |
| ckdyntest `devpersist/remote` (mode) | PASS (copy) | PASS (RDMA) | PASS (IPC) | PASS (IPC) |
| ckdyntest `devpersist/samepe` | PASS (copy) | PASS (copy) | PASS (copy) | PASS (copy) |
| `examples/charm++/cuda/gpudirect/persistent` (mode) | | PASS (RDMA) | PASS (IPC) | PASS (IPC) |
| `device_persistent.py` (mode; cupy device == `charm.myDevice()`) | PASS (copy) | PASS (rdma; PEs 0/0, one GPU each) | PASS (ipc; PEs 0/1) | PASS (ipc; PEs 0/1) |
| `posted_receive.py` (host and device phases) | PASS | PASS | PASS | PASS |
| `device.py` (DEVICE slot mode) | PASS (copy) | PASS (RDMA) | PASS (RDMA) | PASS (IPC) |
| `posted_receive.py -mismatch` | aborts with both lengths | | | |

With the final rule `n2vis` shows IPC for persistent get and put without `+gpushm`
(ckdyntest, the C++ example, `device_persistent.py`); DEVICE slots in `n2vis` stay on
RDMA, since their IPC path is `+gpushm`'s. In `n2` the IPC handle does not open (the
peer's GPU is not visible) and the persistent transfers use RDMA.

Two nodes (job 23677376, gpua001 and gpua009, one A100 each, at 669707835; the RDMA code
is unchanged since, and ran on one node in the `n2` shape after step 4b): ckdyntest all
phases PASS (`devpersist/remote` RDMA), `device_persistent.py`, `posted_receive.py`,
`device.py` PASS.

**`+gpushm`.** The mode is printed once per process: `CkDyn> DEVICE slots: ...` and
`CkDevicePersistent> get/put: ...`. With `+gpushm` and both GPUs visible, DEVICE slots
between the two processes take the IPC path (the first runs of that path). With one GPU
per task, `+gpushm` aborts in `hapiInit` ("Fatal hapi Error invalid argument",
hapi_impl.cpp:1739, `ipcHandleOpen`): HAPI opens the other process's communication
buffer, whose GPU that process cannot see. `+gpushm` therefore needs every GPU of the
node visible to every process.

**Measurements** (16 MB unless stated; one-way, median of 20 after 3 warm-up). DEVICE
slot: `bench_zc_delta.py device`, round trip / 2 between main and element 1. Persistent:
`bench_devpersist.py`, element 1 times the call to the future's value (done callback
included).

One node (job 23677847, gpua node with buses 07:00.0 and 46:00.0; PE 1 on device 1 in
the `n2vis` shapes):

| path | n2 (one GPU per task) | n2vis | n2visshm |
|---|---|---|---|
| DEVICE slot (ms) | 1.246 (RDMA) | 1.250 (RDMA) | 0.266 (IPC) |
| persistent get (ms) | 1.160 (RDMA) | 0.200 (IPC) | 0.199 (IPC) |
| persistent put (ms) | 1.287 (RDMA) | 0.201 (IPC) | 0.200 (IPC) |

Two nodes (job 23677376, Slingshot):

| size | persistent get (ms) | persistent put (ms) | DEVICE slot (ms) |
|---|---|---|---|
| 1 MB | 0.069 | 0.084 | |
| 4 MB | 0.244 | 0.237 | |
| 16 MB | 0.848 | 0.857 | 0.895 |
| 64 MB | 4.085 | 4.147 | |

Answers. Over RDMA (between nodes, or between processes on one node whose GPUs cannot
see each other) the persistent path saves little against a DEVICE slot message at 16 MB:
0.85 against 0.90 ms between nodes, 1.16-1.29 against 1.25 ms on one node; the transfer
dominates, and the message path's registration and descriptors cost under 0.1 ms. On
one node IPC beats RDMA by about 6 times for persistent transfers (0.20 against 1.16-1.29
ms; 84 GB/s between two A100s) and by 4.7 times for DEVICE slots (0.27 against 1.25 ms
with `+gpushm`); there the persistent path saves 0.07 ms (25%) against the DEVICE slot,
whose IPC path stages through the `+gpushm` communication buffer. Between nodes 16 MB
take 0.85 ms (19.7 GB/s), 64 MB 4.1 ms (16.4 GB/s). One-node RDMA times vary between
nodes (0.82 ms on gpua086 in job 23677400, 1.23-1.25 ms on gpua011 and in job 23677847),
so the tables compare paths within one job.

**C++ comparison, no binding (2026-11-12, A40, 741e26982)**

Charm++'s own ping-pong benchmarks `benchmarks/charm++/cuda/gpudirect/latency` and
`latency-persistent`, built inside the job against the charm++-target CUDA reconverse
build, ran with `-s 1024 -x 67108864 -i 1000 -l 100 -w 10 +pe 2` (1000 timed iterations up
to 8 KB, 100 above, 10 warm-up). `latency` without `-z` runs a host-staged phase (D2H copy
and stream sync, a host message, memcpy and H2D copy and sync on the receiver) and then
the `nocopydevice` zero-copy parameter phase; `latency-persistent` does
`CkDevicePersistent::put` into the peer's buffer, whose receive callback sends the reply.
Every number is the mean of (round trip / 2) timed on PE 0, in microseconds; the
persistent number includes the receive callback of each put. One node: job 23841776 on
gpub002 (gpuA40x4-interactive, two NVIDIA A40); the modes printed by
`+gpupersistverbose` are "IPC open ... failed; using RDMA" in `n2` and "CUDA/HIP IPC
between processes on one host" in `n2vis`. The script is tools/delta/cxx-latency.sbatch.

These numbers were measured in C++ with no Python binding involved; the tables above
(`bench_zc_delta.py`, `bench_devpersist.py`, median of 20 on A100 nodes) were measured
through the binding, so they include the Python-side call, future and callback costs and
are not directly comparable cell by cell.

One node, two processes, one GPU per task (`n2`; both paths on RDMA within the host), us:

| size | host-staged message | zero-copy parameter | persistent put |
|---|---|---|---|
| 1 KB | 18.7 | 15.3 | 18.5 |
| 4 KB | 20.4 | 15.3 | 18.3 |
| 16 KB | 37.0 | 17.2 | 18.4 |
| 64 KB | 45.5 | 19.3 | 19.7 |
| 256 KB | 88.9 | 28.0 | 24.4 |
| 1 MB | 248 | 62.8 | 42.0 |
| 4 MB | 882 | 223 | 113 |
| 16 MB | 7960 | 846 | 488 |
| 64 MB | 34500 | 3330 | 1910 |

One node, two processes, both GPUs visible (`n2vis`; persistent on CUDA IPC, the
zero-copy parameter on RDMA, or on HAPI's IPC buffer with `+gpushm`), us:

| size | host-staged message | zero-copy parameter | zero-copy parameter, `+gpushm` | persistent put |
|---|---|---|---|---|
| 1 KB | 19.5 | 15.5 | 25.8 | 8.57 |
| 4 KB | 20.5 | 15.4 | 26.0 | 8.31 |
| 16 KB | 37.5 | 17.3 | 26.5 | 8.59 |
| 64 KB | 48.3 | 19.5 | 28.5 | 9.63 |
| 256 KB | 97.3 | 28.1 | 36.5 | 13.5 |
| 1 MB | 254 | 62.7 | 68.0 | 30.4 |
| 4 MB | 891 | 223 | 214 | 98.9 |
| 16 MB | 7700 | 846 | 766 | 360 |
| 64 MB | 34400 | 3330 | aborted | 1400 |

The `+gpushm` run aborted at 64 MB: "Not enough memory on device communication buffer
(33554432 free)"; HAPI's communication buffer defaults to 64 MB, so a 64 MB message does
not fit. The script now passes `+gpucommbuffer 256` in that shape; that cell has not been
rerun.

Two nodes, one process and one GPU each (RDMA over Slingshot; job 23846960 on gpub074 and
gpub082, one A40 each; `+gpupersistverbose`: "put: RDMA on registered device memory"), us:

| size | host-staged message | zero-copy parameter | persistent put |
|---|---|---|---|
| 1 KB | 19.3 | 15.7 | 18.3 |
| 4 KB | 20.4 | 15.7 | 18.2 |
| 16 KB | 37.8 | 17.9 | 18.5 |
| 64 KB | 49.4 | 21.5 | 19.7 |
| 256 KB | 94.6 | 33.4 | 24.4 |
| 1 MB | 269 | 78.4 | 42.6 |
| 4 MB | 909 | 278 | 112 |
| 16 MB | 7820 | 1050 | 393 |
| 64 MB | 34500 | 4280 | 1400 |

Reading. Below 64 KB the fixed per-message cost dominates every path: the zero-copy
parameter takes 15-19 us, the persistent put 18-20 us over RDMA (no gain, slightly slower
up to 64 KB) and 8.3-9.6 us over CUDA IPC (about 1.8 times faster than the zero-copy
parameter). Over RDMA the persistent put gains from 128 KB on and reaches 1.7 times at
16-64 MB (488 against 846 us at 16 MB; 35 against 20 GB/s at 64 MB); with IPC it is 2.1-2.7
times faster than either zero-copy form from 256 KB up (360 against 846 and 766 us at 16 MB;
48 GB/s at 64 MB). The host-staged message costs only 3-5 us more than the zero-copy
parameter at 1-4 KB, but from 8 KB on it is 2-10 times slower than the zero-copy parameter
and 16-25 times slower than the persistent put at 16-64 MB (7700-7960 us at 16 MB).

Between nodes the persistent put is slower than the zero-copy parameter up to 16 KB (18.2
against 15.7 us), equal at 32 KB, and faster from 64 KB on: 1.8 times at 1 MB, 2.7 times at
16 MB and 3.1 times at 64 MB (1400 against 4280 us). The zero-copy parameter is 1.2-1.3
times slower between nodes than over one-node RDMA from 1 MB up (4280 against 3330 us at
64 MB), while the persistent put takes the same time or less (1400 against 1910 us at
64 MB). These persistent times are not yet explained: 64 MB in 1400 us is 48 GB/s, above
the 25 GB/s of a 200 Gb/s Slingshot port, 16 MB took 0.85 ms through the binding between
two A100 nodes (table above), and `latency-persistent` does not check the received bytes.
A run that checks the destination buffer after each put is needed before these
persistent numbers are used.

**Open.** HAPI's `+gpushm` with one GPU per task (above; the coordinator handles the
issue). The pin in `charm_version.txt` moves at merge (the binding needs
`CkDynUnpostBuffer` and `CkDynHapiUseMyDevice`). The two-node job was not repeated after
step 4b.

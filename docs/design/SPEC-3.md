# SPEC-3: zero-copy host buffers, device buffers, HAPI callbacks

Step 3 of `DESIGN.md` 7.1, written 2026-11-08 after steps 1, 2, 4a-4d, 5a and 6.
Design text: DESIGN 3.14 (slots and C API), 4.3 (argument table), 5.x items 4 and 5,
6.4 (CUDA build), 8.4 (device path never run), 8.7 open item 3 (threshold).
Layer 2 work goes on Charm++ branch `ckdyn-dev`; binding work on charm4py branch
`thin`. Nothing in this step opens a pull request.

## 1. Scope

In: `CKDYN_SLOT_NCPY` (host zero-copy) through the non-post and post paths,
`CKDYN_SLOT_DEVICE`, `CkDynHapiAddCallback`, the C++ test phases, the binding's use of
them (numpy above the threshold, `__cuda_array_interface__` arguments, received device
buffers, `charm.hapiAddCudaCallback`), the Delta recipe and the first GPU run.

Out: a Python-level post API (the post path is exercised only by the C++ test; a
Python surface is a step 7 candidate), zero-copy in broadcasts, multicasts and
reductions (sent as BUFFER copies, DESIGN 3.14), HIP runs (the code is written for
both through `hapi_portable.h`; only CUDA is run), wheels with device support (6.4).

## 2. C API (`ckdyn.h`)

The declarations already in the header stay as they are:

```c
void CkDynMsgAddNcpy(CkDynMsgBuilder *b, const CkDynBufDesc *d, CkDynBufDone done, void *arg);
void CkDynMsgAddDevice(CkDynMsgBuilder *b, const CkDynDevDesc *d, CkDynBufDone done, void *arg);
void CkDynPostBuffer(const CkDynTarget *t, int ep, int slot, const CkDynBufDesc *d);
void CkDynPostDeviceBuffer(const CkDynTarget *t, int ep, int slot, const CkDynDevDesc *d);
void CkDynHapiAddCallback(uint64_t stream, const CkDynCallback *cb);
int  CkDynHasDeviceSupport(void);
```

Semantics:

- **AddNcpy / AddDevice** add a slot of type NCPY or DEVICE. `d->ptr` must stay valid
  until `done(arg)` is called on the sending PE, which happens exactly once per slot,
  after the data has left the source (same-PE delivery included) or when the builder
  is abandoned. `done` is called from the scheduler on the sending PE, never inside
  `CkDynMsgAddNcpy` or `CkDynMsgFinish`. The header's `nbuf` counts these slots.
  `elem`, `ndim`, `shape`, `tag` and `bflags` are recorded in the slot exactly as for
  BUFFER, so a receiver reads NCPY and DEVICE slots with `CkDynViewBufferPtr` and the
  same shape convention.
- **A message with `nbuf > 0` can only be sent point to point** (array element, group
  branch, nodegroup branch, future, callback). `CkDynSend` to a broadcast or section
  target, and `CkDynMsgFinishMcast`, abort with a message naming the slot. The binding
  never builds such a message (section 4).
- **Receive, non-post path:** CkDyn allocates the destination inside the delivered
  message, so the view's pointer for an NCPY slot is owned by the message and lives as
  long as a `CkDynMsgRetain` holds it, the same rule as BUFFER (DESIGN 3.4). For a
  DEVICE slot the destination is device memory from HAPI's pool
  (`hapiPoolMalloc`) and is freed when the message's last retain is released; the
  view's pointer is the device pointer, and the slot carries the device ordinal in a
  field the implementation picks (document it in `ckdyn.h`). The data is complete when
  the hook runs, on every stream.
- **Receive, post path:** `CkDynPostBuffer(t, ep, slot, d)` registers, on the calling
  PE, a destination for the next message to local object `t`, entry `ep`, slot index
  `slot`. The next such message delivers that slot into `d->ptr` (the lengths must
  match, else abort with both lengths), the view's pointer is `d->ptr`, and the post is
  consumed. One post per (object, ep, slot) may be outstanding; a second abort. A post
  never outlives the object: a posted buffer for an element that migrates or is
  destroyed is dropped. `CkDynPostDeviceBuffer` is the same for device memory.
  Implementation note: Charm++'s `CkPostBuffer`/`CkMatchBuffer` tag space is global per
  PE; CkDyn can use it with a tag it mints, or keep its own table and issue the gets
  itself through `CkNcpyBuffer::get`. Either is acceptable; say which in `ckdyn.md`.
- **Wire format.** Reuse the layout the generated marshalling code produces for
  `nocopy` parameters (`impl_num_rdma_fields`, root node, then the pupped
  `CkNcpyBuffer`s; `xi-Parameter.C:388-396`) at the start of `buf`, and mark the
  envelope `CMK_ZC_P2P_SEND_MSG` (or `CMK_ZC_DEVICE_MSG`), so that `_processHandler`'s
  existing `CkRdmaIssueRgets` call (ck.C:1399-1415) and `CkRdmaDeviceIssueRgets` do the
  transfer with no change to ck.C. CkDyn's slot `off` for an NCPY slot is then resolved
  at view time from the rewritten descriptor. If this proves unworkable, a CkDyn-owned
  two-message protocol (descriptor message, receiver issues `CkNcpyBuffer::get`,
  completion callback delivers) is the fallback; record the reason.
- **Same-PE sends:** CkDyn may deliver an NCPY or DEVICE slot by copy when sender and
  receiver are the same PE. `done` is still called from the scheduler.
- **Registration mode** `CK_BUFFER_UNREG` (no registration outlives the message).
- **`CkDynHapiAddCallback(stream, cb)`**: when `stream` reaches the point of the call,
  `cb` is sent an empty `CkDynMsg` (`CkDynCallbackSend` with a zero-slot message) from
  the scheduler of the calling PE. Quiescence accounting as in `hapiAddCallback`.
  In a build without `CMK_CUDA`/`CMK_HIP` it aborts with a message, as now.
- **ABI:** no change to `CkDynHooks`, so `CKDYN_API_VERSION` stays 2. New fields on
  `CkDynDevDesc`, if any, go at the end.

## 3. C++ tests (`tests/charm++/ckdyn/`)

New phases of `ckdyntest`, skippable with `-skip`:

- `ncpy`: 16 MB buffer, array element 0 to element on the last PE and back, non-post
  path; check contents (pattern), `done` called exactly once per send with its `arg`,
  the retained view's pointer valid after the hook returns and freed with the message.
  Then the same through the post path (receiver posts before the send; also post after
  the send has been issued, which must still work when the message arrives later).
  Then a same-PE send. Runs at `+p1`, `+p4` and under `lcrun -n 2`.
- `device` (compiled only under `CMK_CUDA || CMK_HIP`, skipped otherwise): a 16 MB
  device buffer (`cudaMalloc`, filled by `cudaMemcpy` from a host pattern) sent as a
  DEVICE slot to an element on another PE and on the same PE; the receiver copies back
  with `cudaMemcpy` and checks; the message release frees the pool memory (check with
  the pool's statistics or a second allocation of the same size).
- `hapi` (same guard): `cudaMemcpyAsync` on a stream, `CkDynHapiAddCallback` to a
  future, the future's deposit completes the phase.

The Makefile `test` target runs the new phases with the others. The CUDA phases
compile with charmc in a `cuda` build; no `.cu` file (no kernels are needed).

## 4. Binding (`charm4py/src/ckdyn_py.cpp`, `charm4py/charm.py`)

- Sender table (DESIGN 4.3): a contiguous numpy array (or any buffer-protocol object)
  of `nbytes >= charm.options.zerocopy_threshold` goes as NCPY with `done` releasing
  the binding's reference to the object, **except** when the message is a broadcast, a
  section multicast, a reduction contribution or a `Ray`/pool task payload, where it
  stays a BUFFER copy. The threshold is read at pack time, so a test can lower it.
- An argument with `__cuda_array_interface__` goes as DEVICE (`ptr = data[0]`,
  `typestr` to `elem`, `shape`, `strides` must be None or C-contiguous, `stream` from
  the interface if present else 0). Without device support in libcharm:
  `TypeError("device arrays need a CUDA or HIP build of Charm++")`. The reference is
  held until `done`.
- Receive: an NCPY slot becomes a numpy view over the retained message, exactly like
  BUFFER (`copy_received_arrays` applies). A DEVICE slot becomes `_ckdyn.DeviceBuffer`:
  holds a `MsgRef`, implements `__cuda_array_interface__` version 3 (`shape`,
  `typestr`, `data = (ptr, False)`, `strides = None`, `stream = None`), plus `.device`
  (ordinal) and `.nbytes`. `cupy.asarray(buf)`, `torch.as_tensor(buf)` and
  `numba.cuda.as_cuda_array(buf)` then work without a copy.
- Return values and `ret=True` follow the same table.
- `charm.hapiAddCudaCallback(stream, future)` is already routed to
  `_ckdyn.hapi_add_callback`; the empty message deposited into the future becomes
  `None`, as in 1.x.
- Options: `zerocopy_threshold` (exists, 1 MB) and `copy_received_arrays` (exists).

## 5. Python tests (`tests/thin/`)

- `zerocopy.py`: threshold lowered to 64 KB; a 2 MB float64 array sent element to
  element and back at 1 and 2 processes, contents checked, the sender's array still
  intact after the send (the reference is dropped only in `done`), a received view that
  is kept past the method keeps its data, a broadcast of the same array arrives (as a
  copy) on every element. Also a `ret=True` return of a large array.
- `device.py` (skipped without `_ckdyn.has_device_support()` or without cupy): a cupy
  array sent to an element on another PE, received as `DeviceBuffer`, `cupy.asarray`
  and compare; a round trip; `hapiAddCudaCallback` on a cupy stream after a kernel.
- `examples/cuda/hapi/hapi-cuda-callback.py` and `multi_gpu_callback.py` run unchanged
  (numba) on Delta.

## 6. Gates and measurements

Mac (loopback, copy-based `CmiIssueRget`): C++ `ncpy` phases at `+p1`, `+p4`,
`lcrun -n 2`; thin gate stays 100 of 100 plus `zerocopy.py` at 1 and 2 processes.

Delta (`reconverse-linux-x86_64 cuda`, one node, two processes, two A100 or A40):
all `ckdyntest` phases including `device` and `hapi` at `-n 1` and `-n 2` under
`srun --mpi=pmix`; `tests/charm++/cuda/hapitest`; the thin test suite; `device.py`;
the two 1.x hapi examples. Also two nodes, one process each, for `ncpy` and `device`
over the Slingshot fabric (the one-node case uses shared memory or IPC).

Measure and record in this file: one-way time of a 16 MB array as BUFFER and as NCPY
on the Mac (2 processes) and on Delta (2 nodes), and the crossover size on Delta, to
answer DESIGN 8.7 item 3.

Decision F18: a periodic GPU job on Delta. This step delivers the scripts
(`tools/delta/env-cuda.sh`, `build.sh`, `gpu-gate.sbatch`) and the `scrontab` line;
Kale installs the line, since it runs under his allocation.

## 7. Status

Status 2026-11-08 (Mac part done; Delta part not started). Charm++ `ckdyn-dev`
4929702e8, 00b174877 (HAPI callback), 96de91030 (NCPY, posts, DEVICE code),
7dc8e25a0 (ncpy phases), 365a128b5 (device phases); charm4py `thin` 012ff3a
(binding), 80639f5 (tests). Mechanism: `doc/internals/ckdyn.md`, "Zero-copy".

**Built.** Sections 2-5 as specified, except the deviations below. The wire format is
the generated layout (descriptor prefix at the start of `buf`, envelope marked
`CMK_ZC_P2P_SEND_MSG` or `CMK_ZC_DEVICE_MSG`); ck.C and ckrdma.C are unchanged, and
the fallback protocol was not needed. The receiver finds the CkDyn header after the
prefix (`dynPayload`; no prefix when `buf` starts with the magic) and reads a slot's
pointer from its rewritten descriptor. `CkDynSlot.reserved` is a DEVICE slot's device
ordinal, `reserved2` the descriptor index (documented in `ckdyn.h`). Posts: CkDyn's own
per-object table (not `CkPostBuffer`/`CkMatchBuffer`). Binding: the sender table with
the threshold read at pack time; copies kept for broadcasts, sections, reduction
contributions, constructor messages, Ray sends and classes that set
`_ckdyn_no_zerocopy` (the pool's scheduler and workers); `__cuda_array_interface__`
arguments; `_ckdyn.DeviceBuffer`; `ret=True` values through the same table;
`_ckdyn.zerocopy_counts()` for tests. `docs/gpus.rst` rewritten for 2.0.

**Deviations, each kept.**

1. Host posts are applied at delivery with one copy on the receiving PE (from the
   delivered message into the posted buffer; the descriptor is then pointed at it).
   The sender cannot know whether the receiver posted, and the non-post path moves
   the data in `_processHandler` before the target object is known; receiving
   straight into the posted buffer needs `_processHandler` to consult CkDyn before
   `CkRdmaIssueRgets` (a ck.C change, excluded by section 2), or every NCPY message
   to take the two-delivery `P2P_RECV` flow, whose ownership differs between array
   elements (`CkDeliverMessageFree`) and other kinds (`CkDeliverMessageReadonly`).
   Device posts receive directly (the receiver issues the device transfers).
2. Received DEVICE memory comes from `CkDeviceMalloc` (the registered device pool of
   ckrdmadevice.C), not `hapiPoolMalloc`, which allocates pinned host memory. It is
   freed after the hook when the binding kept no retain, else at the release that
   leaves the message's reference count at 1 (process-wide table).
3. Beyond broadcasts and sections, CkDyn also refuses zero-copy slots in multi-branch
   sends, constructor messages, reduction payloads (`CkDynMsgPayload`) and broadcast
   or section callbacks; a message cannot mix NCPY and DEVICE slots (the generated
   code cannot either); at most 4096 zero-copy slots; posts for nodegroup branches
   are refused. A send of a zero-copy message drops `CK_MSG_INLINE` (an inline
   delivery would bypass the transfer). An `IGNORE`/`EXIT` callback frees the message
   (running its done functions) and sends an empty one.
4. Binding: a device array in a copied context (broadcast, section, reduction, Ray,
   pool) is pickled as in 1.x instead of raising; only the missing device support
   raises `TypeError`. A device array to a singleton chare whose PE is not known yet
   (created with `onPE=-1`, not yet constructed) raises `TypeError`, since
   `CkRdmaDeviceOnSender` needs the destination PE. The pool is excluded by a class
   attribute (`_ckdyn_no_zerocopy`), since a pool task is not marked in the message.
5. C++ test: `ncpy/post_after` orders the post after the send deterministically: on
   one PE the post follows the send in the same method; within one process the target
   PE spins until the sender has sent; across processes the post request follows the
   data, which is delivered only after its transfer. The device pool check is "a
   second `CkDeviceMalloc` of the same size returns the same block". `make test` also
   runs `-zcbcast`, which must abort on a broadcast with an NCPY slot.

**Tests run (Mac).** `ckdyntest` with the new phases `ncpy/nonpost`, `ncpy/post_before`,
`ncpy/post_after`, `ncpy/samepe` at `+pe 1`, `+pe 4`, `lcrun -n 2 +pe 2` and
`lcrun -n 2 +pe 4`; the full `make test` sequence passes (randomized queue run,
RotateLB run, restart runs, `+p1`, stale-send and `-zcbcast` aborts); the `device`
and `hapi` phases print SKIP. `tests/thin/zerocopy.py` passes at `+p1` and `+p2`;
`tests/thin/device.py` prints SKIP (no cupy, no device support). Thin gate
(`logs/gate-4c/gate4a.py 1 2`): 100 of 100; `gate4c.sh 2`: 10 of 10
(`logs/step-3/`).

**Measurement (Mac, 2 processes).** 16 MB float64, 20 round trips after 3 warm-up,
one-way time = round trip / 2, median of three runs (`logs/step-3/bench.txt`):

| path | one-way (ms) | min one-way (ms) |
|---|---|---|
| BUFFER (threshold 32 MB) | 2.87 | 2.23 |
| NCPY (threshold 1 MB) | 2.57 | 2.08 |

On this Mac NCPY saves the sender's copy into the message and the receiver's copy
out of the received message is replaced by the copy-based get; the 10% difference
says nothing about a fabric with RDMA. Delta (2 nodes) and the crossover size are
not measured yet.

**Untested here (no GPU, no CUDA toolkit on this Mac).** Every line under
`CMK_CUDA || CMK_HIP`: in `ckdyn.C` `CkDynMsgAddDevice`, the device prefix in
`zcBuildPrefix`, `zcDevicePrepare`, `zcDeviceIssue`, `zcAfterDelivery`,
`zcBeforeRelease`, the device branches of `zcPrefixLen`, `zcFreeUnsent` and
`zcViewPtr`, `CkDynPostDeviceBuffer`, `CkDynHapiAddCallback`; the `device/remote`,
`device/samepe` and `hapi` phases of `ckdyntest.C`; the binding's `add_device` and
`DeviceBuffer`; `tests/thin/device.py`. `ckdyn.C` and `ckdyntest.C` pass a syntax-only
compile with `CMK_CUDA` defined against stub CUDA headers, nothing more. The Delta
part of section 6 (recipe, scripts, `scrontab` line, GPU runs) is not done.

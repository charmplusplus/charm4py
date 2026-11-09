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

(filled in as the work proceeds)

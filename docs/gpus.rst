====
GPUs
====

.. .. contents::


Charm4py 2 works with GPU arrays in two ways: device arrays can be passed as
arguments of entry methods, and the completion of work on a CUDA stream can set
a Charm4py future (``charm.hapiAddCudaCallback``). Both use the Charm++ HAPI
(Hybrid API) and device zero-copy support, so they need a Charm++ built with
``cuda`` (or ``hip``); see `Building with GPU support`_.

``charm4py._ckdyn.has_device_support()`` returns ``True`` when the Charm++
library Charm4py runs on was built with CUDA or HIP.

Device arrays as arguments
--------------------------

Any argument that implements the `CUDA Array Interface
<https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_
(``__cuda_array_interface__``: CuPy arrays, PyTorch CUDA tensors, Numba device
arrays) is sent device to device when the call is point to point (to one
element of an array, one branch of a group, a chare, or as the value sent to a
future, including ``ret=True`` results). Charm++ picks the transfer: a device
copy within a process, CUDA IPC between processes on one node, and GPUDirect
RDMA between nodes. The array must be C-contiguous. The data must be valid on
the stream named in the interface (``stream``; the default stream when absent):
the sender's stream is synchronized, or ordered with an event, before the data
is read.

Charm4py keeps a reference to the argument until the transfer has read it, so
the sender may drop its own reference right after the call, but must not write
to the array until then.

The receiver gets a ``charm4py._ckdyn.DeviceBuffer``: device memory owned by
the received message, freed when the last reference to the buffer (or to an
array made from it) goes away. It implements ``__cuda_array_interface__``
(version 3), so the usual libraries wrap it without a copy:

.. code-block:: python

    import cupy

    class Worker(Chare):

        def recv(self, buf):
            a = cupy.asarray(buf)    # or torch.as_tensor(buf, device='cuda'),
                                     # or numba.cuda.as_cuda_array(buf)
            ...

    # sender
    x = cupy.arange(1 << 20, dtype=cupy.float64)
    workers[3].recv(x)

``DeviceBuffer`` also has ``shape``, ``typestr``, ``nbytes``, ``device`` (the
device ordinal) and ``ptr``.

Broadcasts, section multicasts, reductions and Ray or pool tasks do not use
device transfers: a device array given to them is pickled (copied through the
host), as in Charm4py 1.x. Without device support in Charm++, sending a device
array raises ``TypeError``.

The same rule applies to host arrays: a numpy array of at least
``charm.options.zerocopy_threshold`` bytes (1 MB by default) in a point-to-point
call is transferred by Charm++ from the sender's memory instead of being copied
into the message, and the received array is a view of the data in the message.

Receiving into a posted array: ``Chare.post_buffer``
-----------------------------------------------------

.. code-block:: python

    self.post_buffer(self.method, slot, array)

makes ``array`` the destination of argument ``slot`` (0-based, counting the
method's arguments) of the next message to ``method`` of this object; the method
is then called with ``array`` itself, not a new array. ``array`` is a numpy array
(C-contiguous, writable) or an object with ``__cuda_array_interface__`` (CuPy,
PyTorch, Numba), and the argument that arrives must have the same number of bytes
(the program aborts with both lengths otherwise). A device array is received
directly into the posted memory; a host array costs one copy on the receiving PE,
from the message into the posted array, which still saves allocating a
message-sized array. When the argument arrives as a copy (a host array below
``zerocopy_threshold``, or any array in a broadcast) it is copied into the posted
array, so the method still receives the posted object.

.. code-block:: python

    import cupy

    class Worker(Chare):

        def __init__(self):
            self.halo = cupy.empty(1 << 20, dtype=cupy.float64)

        def start(self):
            self.post_buffer(self.recv_halo, 0, self.halo)

        def recv_halo(self, halo):
            assert halo is self.halo   # the posted array, filled device to device
            ...
            self.post_buffer(self.recv_halo, 0, self.halo)   # for the next message

One post per (method, slot) may be outstanding; a second raises
``Charm4PyError``. Posts are dropped when the object migrates or is deleted (the
next message then allocates as usual). ``post_buffer`` raises on an object that is
not on this PE and on nodegroup branches.

Persistent device buffers: ``charm.DevicePersistent``
-----------------------------------------------------

A persistent buffer wraps a device array once, registers it with the runtime, and
then moves its bytes to or from another persistent buffer with no entry method
and no per-transfer setup:

.. code-block:: python

    p = charm.DevicePersistent(a)    # a: CuPy array, PyTorch CUDA tensor, ...
    h = p.handle                     # small and picklable: send it as an argument
    f = p.get(h_remote)              # copy the remote buffer into a; a Future
    f.get()                          # None once the bytes are in a
    f = p.put(h_remote)              # copy a into the remote buffer
    p.close()

.. code-block:: python

    class Peer(Chare):

        def setup(self, n):
            self.a = cupy.zeros(n, dtype=cupy.float64)
            self.p = charm.DevicePersistent(self.a)
            return self.p.handle

        @coro
        def pull(self, h):
            self.p.get(h).get()      # self.a now holds the other buffer's data
            return float(self.a.sum())

    h0 = peers[0].setup(n, ret=True).get()
    peers[1].setup(n, ret=True).get()
    total = peers[1].pull(h0, ret=True).get()

``get`` and ``put`` create a future, so they are called from a coroutine (a
``@coro`` entry method or the main function). One transfer per buffer may be
outstanding; another ``get`` or ``put`` before its future has a value raises
``Charm4PyError``, and so does ``close``. The array must stay contiguous and alive
until ``close()``; a buffer garbage-collected without ``close()`` is closed with a
``RuntimeWarning``. The data to be read must be complete before the call (work on
the array's stream is ordered before the transfer; work on other streams is not).
``p.mode(h)`` says how a transfer with ``h`` moves the bytes: ``'copy'`` within a
process, ``'ipc'`` between processes on one host when the other buffer's GPU is
visible to this process (``srun --gpus=N --gpu-bind=none`` rather than
``--gpus-per-task=1``), ``'rdma'`` otherwise (GPUDirect RDMA on registered device
memory, also between processes on one host when the other GPU is not visible). The
``+gpushm`` option is not needed for persistent buffers; it selects the IPC path for
device arrays sent as arguments, and it too needs every GPU of the node visible to
every process. A ``put`` between hosts is carried out as a ``get`` by the
destination's PE, so its future too means that the bytes have arrived.

The device of a PE: ``charm.myDevice()``
-----------------------------------------

Charm++ assigns each PE one GPU: with one PE per process and every GPU of a node
visible to every process, process k of the node gets device k.
``charm.myDevice()`` returns its ordinal, or -1 without device support, and before
any Python code runs on a PE Charm4py makes it the thread's current CUDA device, so
CuPy, PyTorch and Numba allocate on it by default:

.. code-block:: python

    import cupy

    class Worker(Chare):

        def __init__(self):
            assert cupy.cuda.Device().id == charm.myDevice()
            self.a = cupy.zeros(1 << 20)   # on this PE's GPU

Completion callbacks: ``charm.hapiAddCudaCallback``
---------------------------------------------------

.. code-block:: python

    def hapiAddCudaCallback(stream, future)

inserts a callback into the CUDA stream ``stream`` (the stream handle as an
integer: ``cupy.cuda.Stream.ptr``, ``numba`` ``stream.handle.value``,
``torch.cuda.Stream.cuda_stream``). When the stream reaches it, ``future`` is
set to ``None`` on the PE that made the call. Quiescence detection treats the
pending callback as outstanding work.

Building with GPU support
-------------------------

Wheels have no device support. A GPU build is a source build against a Charm++
built with ``cuda`` (or ``hip``); the recipe is in :doc:`install` (section
"GPUs (CUDA and HIP)"). In short, for a build of Charm++ by ``pip``:

.. code-block:: shell

   export CHARM_EXTRA_BUILD_OPTS="cuda"
   pip install .

or point ``CHARM4PY_CHARM_BUILD_DIR`` at a Charm++ ``charm4py`` target built
with ``cuda``, as described there.

Example
-------

.. code-block:: python

    from charm4py import charm
    import time
    import numba.cuda as cuda
    import numpy as np

    @cuda.jit
    def elementwise_sum_kernel(x_in, x_out):
        idx = cuda.grid(1)
        if idx < x_in.shape[0]:
            x_out[idx] = x_in[idx] + x_in[idx]

    def main(args):
        N = 1_000_000
        array_size = (N,)

        s = cuda.stream()
        stream_handle = s.handle.value

        A_host = np.arange(N, dtype=np.float32)

        A_gpu = cuda.device_array(array_size, dtype=np.float32, stream=s)
        B_gpu = cuda.device_array(array_size, dtype=np.float32, stream=s)
        A_gpu.copy_to_device(A_host, stream=s)

        threads_per_block = 128
        blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block

        print("Launching kernel and inserting callback...")
        start_time = time.perf_counter()
        elementwise_sum_kernel[blocks_per_grid, threads_per_block, s](A_gpu, B_gpu)

        return_fut = charm.Future()
        charm.hapiAddCudaCallback(stream_handle, return_fut)
        return_fut.get()
        kernel_done_time = time.perf_counter()
        print(f"Callback received, kernel finished in {kernel_done_time - start_time:.6f} seconds.")

        B_host = B_gpu.copy_to_host(stream=s)

        s.synchronize()

        sum_result = np.sum(B_host)
        print(f"Sum of result is {sum_result}")

        charm.exit()

    charm.start(main)

The example inserts a callback into a CUDA stream and waits on the future for
the completion of a Numba kernel. More examples are in ``examples/cuda``.

``examples/cuda/gpudirect/jacobi3d`` is a 3-D Jacobi iteration with numba.cuda
kernels and one block per chare, whose ghost exchange runs in three modes
(``-m``): through pinned host memory, as device arrays sent into posted receives
(``post_buffer``), and as ``DevicePersistent`` puts followed by a notification
message; its README has the commands for one node and for NCSA Delta. Two
benchmarks measure the same paths between two PEs: ``tests/benchmark/pingpong_gpu.py``
(one-way latency and bandwidth of a ping-pong, modes ``staged``, ``device``,
``device-post`` and ``persistent``) and ``tests/benchmark/bandwidth.py`` (a
window of transfers in flight, modes ``staged``, ``device`` and ``persistent``).
``tests/cuda/single_array.py`` and ``tests/cuda/multi_array.py`` check one and
several device arguments per message, with and without posted receives. These
come from Charm4py 1.x (pull request #208), rewritten for this API.

.. note::

    The device paths (device arrays as arguments, ``DeviceBuffer``,
    ``hapiAddCudaCallback``, posted device arrays and ``DevicePersistent``) were
    run on NCSA Delta A40 and A100 nodes with CUDA 13.2, on one node and on two
    nodes; the measurements are in ``docs/design/SPEC-3.md`` and
    ``docs/design/SPEC-7-device.md`` (section 7 of each).

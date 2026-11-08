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

.. note::

    As of November 2026 the device paths of Charm4py 2 (device arrays as
    arguments, ``DeviceBuffer``, ``hapiAddCudaCallback``) are written but have
    not yet been compiled with CUDA or run on a GPU; the first runs are planned
    on NCSA Delta (``docs/design/SPEC-3.md``, section 7).

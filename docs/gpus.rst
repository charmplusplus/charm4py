====
GPUs
====

.. .. contents::


GPUs are supported in Charm4py via the Charm++ HAPI (Hybrid API) interface.
Presently, this support allows asynchronous completion detection of GPU kernels via Charm4py futures, 
using the function ``charm.hapiAddCudaCallback``.

The HAPI Charm4py API is:

.. code-block:: python

    def hapiAddCudaCallback(stream, future)

.. note::

    For now, ``charm.hapiAddCudaCallback`` only supports numba and torch streams as input. This function inserts a callback 
    into the stream such that when the callback is reached, the corresponding Charm4py future is set.

Enabling HAPI
--------
To build Charm4py with HAPI support, add "cuda" to the Charm build options and follow the steps to build Charm4py from source:

.. code-block:: shell

   export CHARM_EXTRA_BUILD_OPTS="cuda"
   pip install .

.. warning:: 

    To ensure that the underlying Charm build has Cuda enabled, remove any pre-existing builds in charm_src/charm before setting the Cuda option and running install.

Examples
--------

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


The above example demonstrates how to use the Charm4py HAPI interface to insert a callback into a CUDA stream and track 
completion of a numba kernel launch.

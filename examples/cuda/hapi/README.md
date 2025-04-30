## Using Charm4py with CUDA

### HAPI CUDA Callback

Example overview

- The example in `hapi-cuda-callback.py` demonstrates usage of addCudaCallback from the Charm++ HAPI library
- addCudaCallback enables an asynchronous mechanism to wait for kernel completion via Charm4py futures
- The example is based around a simple torch kernel.

Usage

- hapiAddCudaCallback requires a cuda stream handle and a future
- access to the Cuda stream handle depends on the Python library being used. For example...
  - using torch: `stream_handle = torch.cuda.Stream().cuda_stream`
  - using numba: `stream_handle = numba.cuda.stream().handle.value`
- currently, the hapiAddCudaCallback is restricted to torch and numba based Cuda streams.

Running example

- If running locally, use:  

$ python3 -m charmrun.start +p<N> hapi-cuda-callback.py  

- If running on a cluster machine with Slurm, use:  

$ srun -n <N> python3 hapi-cuda-callback.py 

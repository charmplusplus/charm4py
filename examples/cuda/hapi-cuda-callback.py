from charm4py import charm
import time
import torch

# using numba requires the following stream handle
# import numba.cuda as cuda
# s = cuda.stream()
# stream_handle = s.handle.value

def main(args):        
    cuda = torch.device('cuda')
    s = torch.cuda.Stream()  # Create a new stream.
    A = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)
   
    print("Starting computation and inserting callback")
    start_time = time.perf_counter()
    with torch.cuda.stream(s):
        B = torch.sum(A)
    
    # create future to track cuda stream
    return_fut = charm.Future()
    stream_handle = s.cuda_stream
    charm.lib.hapiAddCudaCallback(stream_handle, return_fut) 
    
    # other work can be overlapped with kernel here
    
    return_fut.get()
    
    sum = B.cpu().item()
    elapsed = time.perf_counter() - start_time
    print(f"Kernel done in {elapsed} seconds. Sum result is {sum}")
    
    charm.exit()
    

charm.start(main)
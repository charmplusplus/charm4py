'''
Use one process to launch two torch matmul kernels, each on a separate device
A HAPI callback is registered for each kernel
which triggers two different methods
Must run this program with 2 different gpus
'''

from charm4py import charm
import torch

def main(args):

    N=10000

    if torch.cuda.is_available() is not True:
        print("Error: No GPU detected")
        charm.exit()
    if torch.cuda.device_count() < 2:
        print("Error: fewer than 2 GPUs, only " + str(torch.cuda.device_count()) + " gpus found")
        charm.exit()
    
    cuda0 = torch.device('cuda:0') #first device
    cuda1 = torch.device('cuda:1') #second device

    stream0 = torch.cuda.Stream(device=cuda0)
    stream1 = torch.cuda.Stream(device=cuda1)

    #allocate tensors on device 0
    with cuda0:
        a0 = torch.randn(N,N)
        b0 = torch.randn(N,N)
        c0 = torch.mm(a0, b0)
    
    #allocate tensors on device 1
    with cuda1:
        a1 = torch.randn(N,N)
        b1 = torch.randn(N,N)
        c1 = torch.mm(a1, b1)
    
    #create callbacks (should we implement callbacks to entry methods?)
    future0 = charm.Future()
    future1 = charm.Future()
    futures = [future0, future1]
    charm.hapiAddCudaCallback(stream0.cuda_stream, future0)
    charm.hapiAddCudaCallback(stream1.cuda_stream, future1)

    for fut_object in charm.iwait(futures):
        print('One device kernel complete')

    charm.exit()

charm.start(main)

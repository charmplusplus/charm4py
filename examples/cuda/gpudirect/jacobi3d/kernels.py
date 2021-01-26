from numba import cuda
from numba.cuda import blockDim, blockIdx, threadIdx

@cuda.jit(device=True)
def IDX(i,j,k, block_width, block_height):
    return ((block_width+2)*(block_height+2)*(k)+(block_width+2)*(j)+(i))

@cuda.jit
def initKernel(temperature, block_width, block_height, block_depth):
  i = blockDim.x * blockIdx.x + threadIdx.x
  j = blockDim.y * blockIdx.y + threadIdx.y
  k = blockDim.z * blockIdx.z + threadIdx.z

  if i < block_width + 2 and j < block_height + 2 and k < block_depth + 2:
      temperature[IDX(i, j, k, block_width, block_height)] = 0

@cuda.jit
def ghostInitKernel(ghost, ghost_count):
    i = blockDim.x * blockIdx.x + threadIdx.x
    if i < ghost_count:
        ghost[i] = 0

@cuda.jit
def leftBoundaryKernel(temperature, block_width, block_height, block_depth):
  j = blockDim.x * blockIdx.x + threadIdx.x
  k = blockDim.y * blockIdx.y + threadIdx.y
  if j < block_height and k < block_depth:
    temperature[IDX(0,1+j,1+k, block_width, block_height)] = 1;

@cuda.jit
def rightBoundaryKernel(temperature, block_width, block_height, block_depth):
    pass

@cuda.jit
def topBoundaryKernel(temperature, block_width, block_height, block_depth):
    pass

@cuda.jit
def bottomBoundaryKernel(temperature, block_width, block_height, block_depth):
    pass

@cuda.jit
def frontBoundaryKernel(temperature, block_width, block_height, block_depth):
    pass

@cuda.jit
def backBoundaryKernel(temperature, block_width, block_height, block_depth):
    pass


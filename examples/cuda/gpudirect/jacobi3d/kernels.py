from numba import cuda
from numba.cuda import blockDim, blockIdx, threadIdx

TILE_SIZE_3D = 8
TILE_SIZE_2D = 16

LEFT = 0
RIGHT = 1
TOP = 2
BOTTOM = 3
FRONT = 4
BACK = 5
DIR_COUNT = 6

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
        j = blockDim.x*blockIdx.x+threadIdx.x
        k = blockDim.y*blockIdx.y+threadIdx.y
        if j < block_height and k < block_depth:
            temperature[IDX(block_width+1,1+j,1+k, block_width, block_height)] = 1;


@cuda.jit
def topBoundaryKernel(temperature, block_width, block_height, block_depth):
    i = blockDim.x*blockIdx.x+threadIdx.x
    k = blockDim.y*blockIdx.y+threadIdx.y
    if i < block_width and k < block_depth:
        temperature[IDX(1+i,0,1+k, block_width, block_height)] = 1


@cuda.jit
def bottomBoundaryKernel(temperature, block_width, block_height, block_depth):
    i = blockDim.x*blockIdx.x+threadIdx.x
    k = blockDim.y*blockIdx.y+threadIdx.y
    if i < block_width and k < block_depth:
          temperature[IDX(1+i,block_height+1,1+k, block_width, block_height)] = 1

@cuda.jit
def frontBoundaryKernel(temperature, block_width, block_height, block_depth):
    i = blockDim.x*blockIdx.x+threadIdx.x
    j = blockDim.y*blockIdx.y+threadIdx.y
    if i < block_width and j < block_height:
        temperature[IDX(1+i,1+j,0, block_width, block_height)] = 1;


@cuda.jit
def backBoundaryKernel(temperature, block_width, block_height, block_depth):
    i = blockDim.x*blockIdx.x+threadIdx.x
    j = blockDim.y*blockIdx.y+threadIdx.y
    if i < block_width and j < block_height:
          temperature[IDX(1+i,1+j,block_depth+1, block_width, block_height)] = 1

@cuda.jit
def jacobiKernel(temp, new_temp, block_width, block_height, block_depth):
    i = (blockDim.x*blockIdx.x+threadIdx.x)+1
    j = (blockDim.y*blockIdx.y+threadIdx.y)+1
    k = (blockDim.z*blockIdx.z+threadIdx.z)+1

  if (i <= block_width && j <= block_height && k <= block_depth):
      new_temperature[IDX(i,j,k, block_width, block_height)] =
              (temperature[IDX(i,j,k, block_width, block_height)] +
               temperature[IDX(i-1,j,k, block_width, block_height)] +
               temperature[IDX(i+1,j,k, block_width, block_height)] +
               temperature[IDX(i,j-1,k, block_width, block_height)] +
               temperature[IDX(i,j+1,k, block_width, block_height)] +
               temperature[IDX(i,j,k-1, block_width, block_height)] +
               temperature[IDX(i,j,k+1, block_width, block_height)]) *
              0.142857 # equivalent to dividing by 7

@cuda.jit
def leftPackingKernel(temperature, ghost, block_width, block_height, block_depth):
    j = blockDim.x*blockIdx.x+threadIdx.x;
    k = blockDim.y*blockIdx.y+threadIdx.y;
    if j < block_height and k < block_depth:
          ghost[block_height*k+j] =
          temperature[IDX(1,1+j,1+k, block_width, block_height)]

@cuda.jit
def rightPackingKernel(temperature, ghost, block_width, block_height, block_depth):
    j = blockDim.x*blockIdx.x+threadIdx.x
    k = blockDim.y*blockIdx.y+threadIdx.y
    if j < block_height and k < block_depth:
        ghost[block_height*k+j] =
        temperature[IDX(1,1+j,1+k, block_width, block_height)]
  }


@cuda.jit
def topPackingKernel(temperature, ghost, block_width, block_height, block_depth):
    i = blockDim.x*blockIdx.x+threadIdx.x
    k = blockDim.y*blockIdx.y+threadIdx.y
    if i < block_width and k < block_depth:
        ghost[block_width*k+i] =
        temperature[IDX(1+i,1,1+k, block_width, block_height)]

@cuda.jit
def bottomPackingKernel(temperature, ghost, block_width, block_height, block_depth):
    i = blockDim.x*blockIdx.x+threadIdx.x
    k = blockDim.y*blockIdx.y+threadIdx.y
    if i < block_width and k < block_depth:
        ghost[block_width*k+i] =
        temperature[IDX(1+i,block_height,1+k, block_width, block_height)];
  }

@cuda.jit
def frontPackingKernel(temperature, ghost, block_width, block_height, block_depth):
    i = blockDim.x*blockIdx.x+threadIdx.x
    j = blockDim.y*blockIdx.y+threadIdx.y
    if i < block_width and j < block_height:
        temperature[IDX(1+i,1+j,0, block_width, block_height)] =
        ghost[block_width*j+i]

@cuda.jit
def backPackingKernel(temperature, ghost, block_width, block_height, block_depth):
    i = blockDim.x*blockIdx.x+threadIdx.x
    j = blockDim.y*blockIdx.y+threadIdx.y
    if i < block_width and j < block_height:
        temperature[IDX(1+i,1+j,block_depth+1, block_width, block_height)] =
        ghost[block_width*j+i]


@cuda.jit
def leftUnpackingKernel(temperature, ghost, block_width, block_height, block_depth):
    j = blockDim.x*blockIdx.x+threadIdx.x
    k = blockDim.y*blockIdx.y+threadIdx.y
    if j < block_height and k < block_depth:
        temperature[IDX(0,1+j,1+k, block_width, block_height)] = ghost[block_height*k+j]



@cuda.jit
def rightUnpackingKernel(temperature, ghost, block_width, block_height, block_depth):
    j = blockDim.x*blockIdx.x+threadIdx.x
    k = blockDim.y*blockIdx.y+threadIdx.y
    if j < block_height and k < block_depth:
        temperature[IDX(block_width+1,1+j,1+k,  block_width, block_height)] = ghost[block_height*k+j]

@cuda.jit
def topUnpackingKernel(temperature, ghost, block_width, block_height, block_depth):
    i = blockDim.x*blockIdx.x+threadIdx.x
    k = blockDim.y*blockIdx.y+threadIdx.y
    if i < block_width and k < block_depth:
        temperature[IDX(1+i,0,1+k,  block_width, block_height)] = ghost[block_width*k+i]

@cuda.jit
def bottomUnpackingKernel(temperature, ghost, block_width, block_height, block_depth):
    i = blockDim.x*blockIdx.x+threadIdx.x
    k = blockDim.y*blockIdx.y+threadIdx.y
    if i < block_width and k < block_depth:
        temperature[IDX(1+i,block_height+1,1+k,  block_width, block_height)] = ghost[block_width*k+i]

@cuda.jit
def frontUnpackingKernel(temperature, ghost, block_width, block_height, block_depth):
    i = blockDim.x*blockIdx.x+threadIdx.x
    j = blockDim.y*blockIdx.y+threadIdx.y
    if i < block_width and j < block_height:
        temperature[IDX(1+i,1+j,0,  block_width, block_height)] = ghost[block_width*j+i]

@cuda.jit
def backUnpackingKernel(temperature, ghost, block_width, block_height, block_depth):
    i = blockDim.x*blockIdx.x+threadIdx.x
    j = blockDim.y*blockIdx.y+threadIdx.y
    if i < block_width and j < block_height:
        temperature[IDX(1+i,1+j,block_depth+1,  block_width, block_height)] = ghost[block_width*j+i]

def invokeInitKernel(temp_dev_array, block_width, block_height, block_depth, stream):
    block_dim = (TILE_SIZE_3D, TILE_SIZE_3D, TILE_SIZE_3D)
    grid_dim = (((block_width+2)+(block_dim[0]-1))//block_dim[0], # x
      ((block_height+2)+(block_dim[1]-1))//block_dim[1], # y
      ((block_depth+2)+(block_dim[2]-1))//block_dim[2]) # z

    initKernel[grid_dim, block_dim, stream](temp_dev_array,
                                            block_width, block_height,
                                            block_depth)


def invokeGhostInitKernels(ghosts, ghost_counts, stream):
    #TODO: this fn will probably have to change if the ghosts/counts can't
    # be transferred automatically
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dim3
    block_dim = (256, 1, 1)
    dim3 block_dim(256);
    for i in range(len(ghosts)):
        ghost = ghosts[i]
        ghost_count = ghost_counts[i]
        grid_dim = (ghost_count+block_dim[0]-1)//block_dim[0], 1, 1)

        ghostInitKernel[grid_dim, block_dim, stream](ghosts, ghost_count)

def invokeBoundaryKernels(d_temperature, block_width, block_height, block_depth, bounds, stream):
    block_dim = (TILE_SIZE_2D, TILE_SIZE_2D, 1)

    if bounds(LEFT):
        grid_dim = ((block_height+(block_dim.x-1))//block_dim.x,
                    (block_depth+(block_dim.y-1))//block_dim.y, 1)
        leftBoundaryKernel[grid_dim, block_dim, stream](d_temperature,
                                                        block_width,
                                                        block_height,
                                                        block_depth
                                                        )
    if bounds(RIGHT):
        grid_dim = ((block_height+(block_dim.x-1))//block_dim.x,
                    (block_depth+(block_dim.y-1))//block_dim.y, 1)
        rightBoundaryKernel[grid_dim, block_dim, stream](d_temperature,
                                                         block_width,
                                                         block_height,
                                                         block_depth
                                                         )

    if bounds(TOP):
        grid_dim = ((block_width+(block_dim.x-1))//block_dim.x,
                      (block_depth+(block_dim.y-1))//block_dim.y, 1)
        topBoundaryKernel[grid_dim, block_dim, stream](d_temperature,
                                                      block_width,
                                                      block_height,
                                                      block_depth
                                                      )

    if bounds(BOTTOM):
        grid_dim = ((block_width+(block_dim.x-1))//block_dim.x,
                    (block_depth+(block_dim.y-1))//block_dim.y, 1)
        bottomBoundaryKernel[grid_dim, block_dim, stream](d_temperature,
                                                          block_width,
                                                          block_height,
                                                          block_depth
                                                          )

    if bounds(FRONT):
        grid_dim = ((block_width+(block_dim.x-1))//block_dim.x,
                    (block_height+(block_dim.y-1))//block_dim.y, 1)
        frontBoundaryKernel[grid_dim, block_dim, stream](d_temperature,
                                                         block_width,
                                                         block_height,
                                                         block_depth
                                                         )

    if bounds(BACK):
        grid_dim = ((block_width+(block_dim.x-1))//block_dim.x,
                    (block_height+(block_dim.y-1))//block_dim.y, 1)
        backBoundaryKernel[grid_dim, block_dim, stream](d_temperature,
                                                        block_width,
                                                        block_height,
                                                        block_depth
                                                        )


def invokeJacobiKernel(d_temperature, d_new_temperature, block_width, block_height, block_depth, stream):
    block_dim = (TILE_SIZE_3D, TILE_SIZE_3D, TILE_SIZE_3D)
    grid_dim = ((block_width+(block_dim.x-1))//block_dim.x,
                (block_height+(block_dim.y-1))//block_dim.y,
                (block_depth+(block_dim.z-1))//block_dim.z)

    jacobiKernel[grid_dim, block_dim, stream](d_temperature,
                                              d_new_temperature,
                                              block_width,
                                              block_height,
                                              block_depth
                                              )


def inbokePackingKernel(d_temperature, d_ghost, dir, block_width, block_height, block_depth, stream):
    block_dim = (TILE_SIZE_2D, TILE_SIZE_2D, 1)

    if dir == LEFT:
        grid_dim = ((block_height+(block_dim.x-1))//block_dim.x,
                    (block_depth+(block_dim.y-1))//block_dim.y, 1)
        leftPackingKernel[grid_dim, block_dim, stream](d_temperature,
                                                       d_ghost,
                                                       block_width,
                                                       block_height,
                                                       block_depth
                                                       )
    elif dir == RIGHT:
        grid_dim = ((block_height+(block_dim.x-1))//block_dim.x,
                    (block_depth+(block_dim.y-1))//block_dim.y, 1)
        rightPackingKernel[grid_dim, block_dim, stream](d_temperature,
                                                        d_ghost,
                                                        block_width,
                                                        block_height,
                                                        block_depth
                                                        )
    elif dir == TOP:
        grid_dim = ((block_width+(block_dim.x-1))//block_dim.x,
                    (block_depth+(block_dim.y-1))//block_dim.y, 1)
        topPackingKernel[grid_dim, block_dim, stream](d_temperature,
                                                      d_ghost,
                                                      block_width,
                                                      block_height,
                                                      block_depth
                                                      )
    elif dir == BOTTOM:
        grid_dim = ((block_width+(block_dim.x-1))//block_dim.x,
                    (block_depth+(block_dim.y-1))//block_dim.y, 1)
        bottomPackingKernel[grid_dim, block_dim, stream](d_temperature,
                                                         d_ghost,
                                                         block_width,
                                                         block_height,
                                                         block_depth
                                                         )
    elif dir == FRONT:
        grid_dim = ((block_width+(block_dim.x-1))//block_dim.x,
                    (block_height+(block_dim.y-1))//block_dim.y, 1)
        frontPackingKernel[grid_dim, block_dim, stream](d_temperature,
                                                        d_ghost,
                                                        block_width,
                                                        block_height,
                                                        block_depth
                                                        )
    elif dir == BACK:
        grid_dim = ((block_width+(block_dim.x-1))//block_dim.x,
                    (block_height+(block_dim.y-1))//block_dim.y, 1)
        backPackingKernel[grid_dim, block_dim, stream](d_temperature,
                                                       d_ghost,
                                                       block_width,
                                                       block_height,
                                                       block_depth
                                                       )

def invokeUnpackingKernel(d_temperature, d_ghost, dir, block_width, block_height, block_depth, stream):
    block_dim = (TILE_SIZE_2D, TILE_SIZE_2D, 1)

    if dir == LEFT:
        grid_dim = ((block_height+(block_dim.x-1))//block_dim.x,
                    (block_depth+(block_dim.y-1))//block_dim.y, 1)
        leftUnpackingKernel[grid_dim, block_dim, stream](d_temperature,
                                                         d_ghost,
                                                         block_width,
                                                         block_height,
                                                         block_depth
                                                         )
    if dir == RIGHT:
        grid_dim = ((block_height+(block_dim.x-1))//block_dim.x,
                    (block_depth+(block_dim.y-1))//block_dim.y, 1)
        rightUnpackingKernel[grid_dim, block_dim, stream](d_temperature,
                                                          d_ghost,
                                                          block_width,
                                                          block_height,
                                                          block_depth
                                                          )
    if dir == TOP:
        grid_dim = ((block_width+(block_dim.x-1))//block_dim.x,
                    (block_depth+(block_dim.y-1))//block_dim.y, 1)
        topUnpackingKernel[grid_dim, block_dim, stream](d_temperature,
                                                        d_ghost,
                                                        block_width,
                                                        block_height,
                                                        block_depth
                                                        )
    if dir == BOTTOM:
        grid_dim = ((block_width+(block_dim.x-1))//block_dim.x,
                    (block_depth+(block_dim.y-1))//block_dim.y, 1)
        bottomUnpackingKernel[grid_dim, block_dim, stream](d_temperature,
                                                           d_ghost,
                                                           block_width,
                                                           block_height,
                                                           block_depth
                                                           )
    if dir == FRONT:
        grid_dim = ((block_width+(block_dim.x-1))//block_dim.x,
                    (block_height+(block_dim.y-1))//block_dim.y, 1)
        frontUnpackingKernel[grid_dim, block_dim, stream](d_temperature,
                                                          d_ghost,
                                                          block_width,
                                                          block_height,
                                                          block_depth
                                                          )
    if dir == BACK:
        grid_dim = ((block_width+(block_dim.x-1))//block_dim.x,
                    (block_height+(block_dim.y-1))//block_dim.y, 1)
        backUnpackingKernel[grid_dim, block_dim, stream](d_temperature,
                                                         d_ghost,
                                                         block_width,
                                                         block_height,
                                                         block_depth
                                                         )

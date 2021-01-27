from charm4py import *
import array
from numba import cuda
import numpy as np
import kernels

def getArrayAddress(arr):
    return arr.__cuda__array_interface__['data'][0]

def getArraySize(arr):
    return arr.nbytes

def getArrayData(arr):
    return (getArrayAddress(arr), getArraySize(arr))

class Block(Chare):
    def __init__(self, init_done_future):
        self.my_iter = 0
        self.neighbors = 0
        self.remote_count = 0
        self.x = self.thisIndex[0]
        self.y = self.thisIndex[1]
        self.z = self.thisIndex[2]
        self.ghost_sizes = (x_surf_size, x_surf_size,
                            y_surf_size, y_surf_size,
                            z_surf_size, z_surf_size
                            )

        self.ghost_counts = (x_surf_count, x_surf_count,
                             y_surf_count, y_surf_count,
                             z_surf_count, z_surf_count
                             )

        self.bounds = [False] * kernels.DIR_COUNT

        empty = lambda x: [0] * x

        self.h_temperature = None
        self.d_temperature = None
        self.d_new_temperature = None
        self.h_ghosts = empty(kernels.DIR_COUNT)
        self.d_ghosts = empty(kernels.DIR_COUNT)
        self.d_send_ghosts = empty(kernels.DIR_COUNT)
        self.d_recv_ghosts = empty(kernels.DIR_COUNT)
        self.d_ghosts_addr = empty(kernels.DIR_COUNT)
        self.d_send_ghosts_addr = empty(kernels.DIR_COUNT)
        self.d_recv_ghosts_addr = empty(kernels.DIR_COUNT)
        self.d_send_ghosts_size = empty(kernels.DIR_COUNT)
        self.d_recv_ghotss_size = empty(kernels.DIR_COUNT)

        self.stream = cuda.default_stream()

        self.init()

        self.reduce(init_done_future)

    def init(self):
        self.init_bounds(self.x, self.y, self.z)
        self.init_device_data()

    def init_device_data(self):
        temp_size = (block_width+2) * (block_height+2) * (block_depth+2)
        self.h_temperature = cuda.pinned_array(temp_size, dtype=np.float64)
        self.d_temperature = cuda.device_array(temp_size, dtype=np.float64)
        self.d_new_temperature = cuda.device_array(temp_size, dtype=np.float64)

        if use_zerocopy:
            for i in range(kernels.DIR_COUNT):
                self.d_send_ghosts[i] = cuda.device_array(self.ghost_sizes[i],
                                                          dtype=np.float64
                                                          )
                self.d_recv_ghosts[i] = cuda.device_array(self.ghost_sizes[i],
                                                          dtype=np.float64
                                                          )

                d_send_data = getArrayData(d_send_ghosts)
                d_recv_data = getArrayData(d_send_ghosts)

                d_send_addr = array.array('L', [d_send_data[0]])
                d_recv_addr = array.array('L', [d_recv_data[0]])

                d_send_size = array.array('L', [d_send_data[1]])
                d_recv_size = array.array('L', [d_recv_data[1]])

                self.d_send_ghosts_addr[i] = d_send_addr
                self.d_recv_ghosts_addr[i] = d_recv_addr

                self.d_send_ghosts_size[i] = d_send_size
                self.d_recv_ghosts_size[i] = d_recv_size
        else:
            for i in range(kernels.DIR_COUNT):
                self.h_ghosts[i] = cuda.pinned_array(self.ghost_sizes[i],
                                                     dtype=np.float64
                                                     )
                self.d_ghosts[i] = cuda.device_array(self.ghost_sizes[i],
                                                     dtype=np.float64
                                                     )

        kernels.invokeInitKernel(self.d_temperature, block_width, block_height, block_depth,
                                 self.stream
                                 )
        kernels.invokeInitKernel(self.d_new_temperature, block_width, block_height, block_depth,
                                 self.stream
                                 )
        if use_zerocopy:
            kernels.invokeGhostInitKernels(self.d_send_ghosts,
                                           self.ghost_counts,
                                           self.stream
                                           )
            kernels.invokeGhostInitKernels(self.d_recv_ghosts,
                                           self.ghost_counts,
                                           self.stream
                                           )
        else:
            kernels.invokeGhostInitKernels(self.d_ghosts,
                                           self.ghost_counts,
                                           self.stream
                                           )
            for i in range(kernels.DIR_COUNT):
                self.h_ghosts[i].fill(0)

        kernels.invokeBoundaryKernels(self.d_temperature,
                                      block_width,
                                      block_height,
                                      block_depth,
                                      self.bounds,
                                      self.stream
                                      )
        kernels.invokeBoundaryKernels(self.d_new_temperature,
                                      block_width,
                                      block_height,
                                      block_depth,
                                      self.bounds,
                                      self.stream
                                      )
        self.stream.synchronize()


    def init_bounds(self, x, y, z):
        neighbors = 0

        if x == 0:
            self.bounds[kernels.LEFT] = True
        else:
            neighbors += 1
        if x == n_chares_x - 1:
            self.bounds[kernels.RIGHT] = True
        else:
            neighbors += 1
        if y == 0:
            self.bounds[kernels.TOP] = True
        else:
            neighbors += 1
        if y == n_chares_y - 1:
            self.bounds[kernels.BOTTOM] = True
        else:
            neighbors += 1
        if z == 0:
            self.bounds[kernels.FRONT] = True
        else:
            neighbors += 1
        if z == n_chares_z - 1:
            self.bounds[kernels.BACK] = True
        else:
            neighbors += 1

        self.neighbors = neighbors

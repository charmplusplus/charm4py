from charm4py import *
import array
from numba import cuda
import numpy as np
import time
import kernels

def getArrayAddress(arr):
    return arr.__cuda_array_interface__['data'][0]

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

        self.neighbor_channels = empty(kernels.DIR_COUNT)
        self.acive_neighbor_channels = None

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
        self.d_recv_ghosts_size = empty(kernels.DIR_COUNT)

        self.stream = cuda.default_stream()

        self.init()

        self.reduce(init_done_future)

    def init(self):
        self.init_bounds(self.x, self.y, self.z)
        self.init_device_data()
        self.init_neighbor_channels()

    def init_neighbor_channels(self):
        n_channels = self.neighbors
        active_neighbors = []

        if not self.bounds[kernels.LEFT]:
            new_c = Channel(self, self.thisProxy[(self.x-1, self.y, self.z)])
            self.neighbor_channels[kernels.LEFT] = new_c
            # NOTE: we are adding the member 'recv_direction' to this channel!!!
            new_c.recv_direction = kernels.LEFT
            active_neighbors.append(new_c)

        if not self.bounds[kernels.RIGHT]:
            new_c = Channel(self, self.thisProxy[(self.x+1, self.y, self.z)])
            self.neighbor_channels[kernels.RIGHT] = new_c
            new_c.recv_direction = kernels.RIGHT
            active_neighbors.append(new_c)

        if not self.bounds[kernels.TOP]:
            new_c = Channel(self, self.thisProxy[(self.x, self.y-1, self.z)])
            self.neighbor_channels[kernels.TOP] = new_c
            new_c.recv_direction = kernels.TOP
            active_neighbors.append(new_c)

        if not self.bounds[kernels.BOTTOM]:
            new_c = Channel(self, self.thisProxy[(self.x, self.y+1, self.z)])
            self.neighbor_channels[kernels.BOTTOM] = new_c
            new_c.recv_direction = kernels.BOTTOM
            active_neighbors.append(new_c)

        if not self.bounds[kernels.FRONT]:
            new_c = Channel(self, self.thisProxy[(self.x, self.y, self.z-1)])
            self.neighbor_channels[kernels.FRONT] = new_c
            new_c.recv_direction = kernels.FRONT
            active_neighbors.append(new_c)

        if not self.bounds[kernels.BACK]:
            new_c = Channel(self, self.thisProxy[(self.x, self.y, self.z+1)])
            self.neighbor_channels[kernels.BACK] = new_c
            new_c.recv_direction = kernels.BACK
            active_neighbors.append(new_c)

        self.active_neighbor_channels = active_neighbors

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

                d_send_data = getArrayData(self.d_send_ghosts[i])
                d_recv_data = getArrayData(self.d_send_ghosts[i])

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


    @coro
    def sendGhosts(self):
        for dir in range(kernels.DIR_COUNT):
            if not self.bounds[dir]:
                self.sendGhost(dir)

    def updateAndPack(self):
        kernels.invokeJacobiKernel(self.d_temperature,
                                   self.d_new_temperature,
                                   block_width,
                                   block_height,
                                   block_depth,
                                   self.stream
                                   )

        for i in range(kernels.DIR_COUNT):
            if not self.bounds[i]:
                ghosts = self.d_send_ghosts[i] if use_zerocopy else self.d_ghosts[i]

                kernels.invokePackingKernel(self.d_temperature,
                                            ghosts,
                                            i,
                                            block_width,
                                            block_height,
                                            block_depth,
                                            self.stream
                                            )
                if not use_zerocopy:
                    # TODO: change this to the CUDA hooks in charmlib
                    self.d_ghosts[i].copy_to_host(self.h_ghosts[i])
        self.stream.synchronize()

    @coro
    def sendGhost(self, direction):
        send_ch = self.neighbor_channels[direction]

        if use_zerocopy:
            send_ch.send(gpu_src_ptrs = self.d_send_ghosts_addr[direction],
                         gpu_src_sizes = self.d_send_ghosts_size[direction]
                         )
        else:
            send_ch.send(self.h_ghosts[direction])

    @coro
    def recvGhosts(self):
        for ch in charm.iwait(self.active_neighbor_channels):
            # remember: we set 'recv_direction' member
            # directly in the initialization phase
            neighbor_idx = ch.recv_direction

            if use_zerocopy:
                ch.recv(post_buf_addresses = self.d_recv_ghosts_addr[neighbor_idx],
                        post_buf_sizes = self.d_recv_ghosts_size[neighbor_idx]
                        )
                recv_ghost = self.d_recv_ghosts[neighbor_idx]
            else:
                self.h_ghosts[neighbor_idx] = ch.recv()
                self.d_ghosts[neighbor_idx].copy_to_device(self.h_ghosts[neighbor_idx],
                                                           stream=self.stream
                                                           )
                recv_ghost = self.d_ghosts[neighbor_idx]

            kernels.invokeUnpackingKernel(self.d_temperature,
                                          recv_ghost,
                                          ch.recv_direction,
                                          block_width,
                                          block_height,
                                          block_depth,
                                          self.stream
                                          )
        self.stream.synchronize()

    @coro
    def exchangeGhosts(self):
        self.d_temperature, self.d_new_temperature = \
            self.d_new_temperature, self.d_temperature

        self.sendGhosts()
        self.recvGhosts()

    @coro
    def run(self, done_future):
        tstart = time.time()
        comm_time = 0
        for current_iter in range(n_iters + warmup_iters):
            if current_iter == warmup_iters:
                tstart = time.time()

            self.my_iter = current_iter
            self.updateAndPack()

            comm_start_time = time.time()

            self.exchangeGhosts()

            if current_iter >= warmup_iters:
                comm_time += time.time() - comm_start_time


        tend = time.time()

        if self.thisIndex == (0, 0, 0):
            elapsed_time = tend-tstart
            print(f'Elapsed time: {round(elapsed_time,3)} s')
            print(f'Average time per iteration: {round(((elapsed_time/n_iters)*1e3),3)} ms')
            print(f'Communication time per iteration: {round(((comm_time/n_iters)*1e3),3)} ms')
        self.reduce(done_future)

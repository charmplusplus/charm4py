from charm4py import *
import kernels

class Block(Chare):
    def __init__(self, init_done_future):
        self.my_iter = 0
        self.neighbors = 0
        self.remote_count = 0
        self.x = self.thisIndex[0]
        self.y = self.thisIndex[1]
        self.z = self.thisIndex[2]

        self.bounds = [False] * kernels.DIR_COUNT
        self.init_bounds(self.x, self.y, self.z)

        self.h_ghosts = []
        self.d_ghosts = []
        self.d_send_ghosts = []
        self.d_recv_ghosts = []
        self.d_ghosts_addr = []
        self.d_send_ghosts_addr = []
        self.d_recv_ghosts_addr = []

        self.reduce(init_done_future)

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

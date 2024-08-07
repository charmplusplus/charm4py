from charm4py import charm, Chare, Array, Future, coro, Channel
import time
import numpy as np

try:
    from numba import njit
except ImportError:
    # create a dummy numba.njit decorator
    def njit(func):
        return func

@njit
def matmul(C, A, B):
    C += A @ B

class SubMatrix(Chare):
    def __init__(self, subdim_size, charedim, init_done):
        self.subdim_size = subdim_size
        self.charedim = charedim

        self.neighbor_cache = {}

        self.sub_a = np.ones((subdim_size, subdim_size), dtype=np.float64)
        self.sub_a[:,:] = (charedim*self.thisIndex[1]) + self.thisIndex[0]
        self.sub_b = np.ones((subdim_size, subdim_size), dtype=np.float64)
        self.sub_b[:,:] = (charedim*self.thisIndex[0]) + self.thisIndex[1]

        self.recv_a = np.ndarray((subdim_size,subdim_size), dtype=np.float64)
        self.recv_b = np.ndarray((subdim_size,subdim_size), dtype=np.float64)

        self.sub_c = np.zeros((subdim_size, subdim_size), dtype=np.float64)

        warmup_c = np.zeros((subdim_size, subdim_size), dtype=np.float64)

        # ensure the kernel is compiled
        matmul(warmup_c, self.sub_a, self.sub_b)

        self.reduce(init_done)

    def get_neighbor_channel(self, target_idx):
        if target_idx not in self.neighbor_cache:
            self.neighbor_cache[target_idx] = Channel(self,
                                                      self.thisProxy[target_idx]
                                                      )
        return self.neighbor_cache[target_idx]

    @coro
    def cannons_multiplication(self, mult_done_future):
        # do initial shift
        # left-shift
        if self.thisIndex[0] > 0:
            self.shift(0, self.thisIndex[0])
            self.sub_a, self.recv_a = self.recv_a, self.sub_a

        # up-shift
        if self.thisIndex[1] > 0:
            self.shift(self.thisIndex[1], 0)
            self.sub_b, self.recv_b = self.recv_b, self.sub_b

        # todo multiplication kernel, will be interesting to see how they compare
        matmul(self.sub_c, self.sub_a, self.sub_b)

        for iter in range(self.charedim - 1):
            self.shift(0, 1)
            self.shift(1, 0)

            self.sub_a, self.recv_a = self.recv_a, self.sub_a
            self.sub_b, self.recv_b = self.recv_b, self.sub_b

            matmul(self.sub_c, self.sub_a, self.sub_b)

        self.reduce(mult_done_future)

    # the communication routines should be optimized so both sends/receives can complete in parallel
    def shift(self, up_shift, left_shift):
        send_target_idx = ((self.thisIndex[0] - up_shift) % self.charedim,
                           (self.thisIndex[1] - left_shift) % self.charedim
                           )
        recv_target_idx = ((self.thisIndex[0] + up_shift) % self.charedim,
                           (self.thisIndex[1] + left_shift) % self.charedim
                           )

        send_ch = self.get_neighbor_channel(send_target_idx)
        recv_ch = self.get_neighbor_channel(recv_target_idx)

        if left_shift:
            send_ch.send(self.sub_a)
            self.recv_a = recv_ch.recv()
        if up_shift:
            send_ch.send(self.sub_b)
            self.recv_b = recv_ch.recv()


def main(args):
    if len(args) < 3:
        print(f"USAGE: {args[0]} matrix_dim chare_dim")
        print("matrix_dim and chare_dim must be perfect squares "
              "where matrix_dim is divisible by chare_dim"
              )
        charm.exit(1)
    matrix_dim = int(args[1])
    chare_dim = int(args[2])

    if matrix_dim % chare_dim:
        print("ERROR: Matrix dim must evenly divide chare dim.")
        charm.exit(1)

    # size of each chare's sub-matrix
    subdim_size = matrix_dim // chare_dim
    print(f"Size of each chare's sub-array: {8*(subdim_size**2)/(1024**2)}MiB")

    init_done = Future()
    chares = Array(SubMatrix, (chare_dim, chare_dim),
                   args=[subdim_size, chare_dim, init_done]
                   )
    init_done.get()

    mult_done_future = Future()
    tstart = time.time()
    chares.cannons_multiplication(mult_done_future)
    mult_done_future.get()
    tend = time.time()

    print(f"Elapsed time: {tend-tstart}")
    charm.exit()

charm.start(main)

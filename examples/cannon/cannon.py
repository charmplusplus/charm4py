from charm4py import charm, Chare, Group, Array, Future, coro, Channel, Reducer
import time
import numpy as np
from math import sqrt

try:
    from numba import jit
    numbaFound = True
except ImportError:
    numbaFound = False
    # create a dummy numba.jit decorator
    def jit(*args, **kwargs):
        def deco(func):
            return func
        return deco
    njit = jit


class SubMatrix(Chare):
    def __init__(self, subdim_size, charedim, init_done):
        self.subdim_size = subdim_size
        self.charedim = charedim

        self.neighbor_cache = {}

        self.sub_a = np.ones((subdim_size, subdim_size), dtype=np.float64)
        self.sub_a[:,:] = (charedim*self.thisIndex[1]) + self.thisIndex[0]
        self.sub_b = np.ones((subdim_size, subdim_size), dtype=np.float64)
        self.sub_b[:,:] = (charedim*self.thisIndex[0]) + self.thisIndex[1]

        self.recv_a = np.ndarray((subdim_size, subdim_size), dtype=np.float64)
        self.recv_b = np.ndarray((subdim_size, subdim_size), dtype=np.float64)

        self.sub_c = np.zeros((subdim_size, subdim_size), dtype=np.float64)

        if self.thisIndex == (0,0):
            self.whole_a = None
            self.whole_b = None
            self.whole_c = None

        if verify:
            self.reduce(self.thisProxy[(0,0)].display, self.sub_a, Reducer.gather)
            self.reduce(self.thisProxy[(0,0)].display, self.sub_b, Reducer.gather)

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
        self.sub_c = self.sub_a @ self.sub_b

        for iter in range(self.charedim - 1):
            self.shift(0, 1)
            self.shift(1, 0)

            self.sub_a, self.recv_a = self.recv_a, self.sub_a
            self.sub_b, self.recv_b = self.recv_b, self.sub_b

            self.sub_c += self.sub_a @ self.sub_b


        if verify:
            self.reduce(self.thisProxy[(0,0)].display, self.sub_c, Reducer.gather)
        self.reduce(mult_done_future)

    def display(self, matr):
        ls = list()
        for i in range(int(len(matr)**0.5)):
            l = list()
            for j in range(int(len(matr)**0.5)):
                l.append(matr[i*(int(len(matr)**0.5))+j])
            ls.append(l)
        nd = np.block(ls)
        if self.whole_a is None:
            self.whole_a = nd
        elif self.whole_b is None:
            self.whole_b = nd
        elif self.whole_c is None:
            self.whole_c = nd
        if not(self.whole_a is None or self.whole_b is None or self.whole_c is None):
            self.whole_c = self.whole_a @ self.whole_b
            try:
                assert np.allclose(nd, self.whole_c)
            except AssertionError:
                print("ERROR: computed output matrix does not match expected")

    # the communication routines should be optimized so both sends/receives can complte in parallel
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
        print(f"USAGE: {args[0]} matrix_dim chare_dim verify")
    matrix_dim = int(args[1])
    chare_dim = int(args[2])
    verify = False

    if len(args) == 4:
        verify = int(args[3])

    assert (chare_dim) % 1 == 0
    matr_x, matr_y = matrix_dim, matrix_dim
    if matrix_dim ** 2 % chare_dim:
        print("ERROR: Matrix dim must evenly divide chare dim.")
        charm.exit(1)

    charm.thisProxy.updateGlobals({'verify':verify}, awaitable=True).get()
    # size of each chare's sub-matrix
    subdim_size = matrix_dim / (chare_dim)
    assert subdim_size % 1 == 0
    subdim_size = int(subdim_size)

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

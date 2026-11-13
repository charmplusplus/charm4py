"""Several device arrays in one message (from charm4py PR #208, ported to the
Charm4py 2 API). Needs numba (numba-cuda), a GPU and a CUDA build of Charm++;
prints SKIP and exits 0 otherwise.

Run: python -m charmrun.start +p2 tests/cuda/multi_array.py

Element 1 (PE 1) sends element 0 (PE 0) one message with an int, a host numpy
array of 42s, a 16 MB int8 device array of ones and a 16 MB float32 device array
of shape (1024, 4096) holding 0, 1, 2, ...; element 0 checks every argument
(values, shape and type of the received device arrays) and replies. Each exchange
runs with a chare array and with a group, without posts (both device arguments
arrive as charm4py.DeviceBuffer) and with both device arguments posted
(Chare.post_buffer on argument 2 and argument 3: they arrive in the posted numba
arrays). PR #208 sent 1 GB arrays; 16 MB keeps the test short.
"""
from charm4py import charm, Chare, Array, Group, Future, coro
from charm4py import _ckdyn
import numpy as np

try:
    from numba import cuda
except ImportError:
    cuda = None

N = 16 << 20
SHAPE2 = (1024, N // 4 // 1024)


def host1():
    return np.ones(N, dtype=np.int8)


def host2():
    return np.arange(N // 4, dtype=np.float32).reshape(SHAPE2)


class A(Chare):

    def __init__(self):
        self.idx = self.thisIndex[0] if isinstance(self.thisIndex, tuple) else self.thisIndex
        self.partner = self.thisProxy[1 - self.idx] if self.idx < 2 else None
        self.p1 = self.p2 = None

    def post(self):
        self.p1 = cuda.device_array(N, dtype=np.int8)
        self.p2 = cuda.device_array(SHAPE2, dtype=np.float32)
        self.post_buffer(self.recv, 2, self.p1)
        self.post_buffer(self.recv, 3, self.p2)

    def recv(self, i, h, d1, d2, posted, done):
        """ element 0 """
        assert i == 20
        assert np.array_equal(h, np.full(100, 42, dtype=np.int32)), 'host argument differs'
        if posted:
            assert d1 is self.p1 and d2 is self.p2, 'device arguments are not the posted arrays'
        else:
            for d, shape, ts in ((d1, (N,), '|i1'), (d2, SHAPE2, '<f4')):
                assert isinstance(d, _ckdyn.DeviceBuffer), type(d)
                assert d.shape == shape and d.typestr == ts, (d.shape, d.typestr)
            d1, d2 = cuda.as_cuda_array(d1), cuda.as_cuda_array(d2)
        ok = np.array_equal(d1.copy_to_host(), host1()) and np.array_equal(d2.copy_to_host(), host2())
        done.send(ok)

    @coro
    def run(self, posted):
        """ element 1 """
        if posted:
            self.partner.post(ret=True).get()
        d1 = cuda.to_device(host1())
        d2 = cuda.to_device(host2())
        done = Future()
        self.partner.recv(20, np.full(100, 42, dtype=np.int32), d1, d2, posted, done)
        return done.get()


def main(args):
    if cuda is None or not cuda.is_available() or not _ckdyn.has_device_support():
        why = ('numba is not installed' if cuda is None else 'no GPU is visible to numba'
               if not cuda.is_available() else 'this Charm++ has no CUDA or HIP support')
        print('SKIP multi_array.py: ' + why)
        charm.exit()
        return
    if charm.numPes() < 2:
        print('SKIP multi_array.py: needs 2 PEs')
        charm.exit()
        return
    for kind, chares in (('array', Array(A, 2)), ('group', Group(A))):
        charm.awaitCreation(chares)
        for posted in (False, True):
            assert chares[1].run(posted, ret=True).get() is True, '%s posted=%s: data differs' % (kind, posted)
        print('multi_array: %s passed (two DeviceBuffers, two posted receives)' % kind)
    charm.exit()


charm.start(main)

"""One device array per message (from charm4py PR #208, ported to the Charm4py 2 API).
Needs numba (numba-cuda), a GPU and a CUDA build of Charm++; prints SKIP and exits 0
otherwise.

Run: python -m charmrun.start +p2 tests/cuda/single_array.py

Element 1 (PE 1) sends element 0 (PE 0) the arguments (1, 2, "hello", a 1 MB numpy
array of ones, a 1 MB numba device array of fives); element 0 checks them and sends
the received device array back with (2, 3); element 1 checks it. Each exchange runs:
- with a chare array and with a group;
- without posts: the device argument arrives as a charm4py.DeviceBuffer, wrapped
  with numba.cuda.as_cuda_array;
- with posts (Chare.post_buffer): the device argument arrives in the numba array
  the receiver posted, the same object (PR #208's addr_optimization, which posted
  receive addresses on a channel);
- over a Channel: the device array is an argument of the channel's message, so it
  is sent device to device too and arrives as a DeviceBuffer (channels have no
  posted receives).
"""
from charm4py import charm, Chare, Array, Group, Future, Channel, coro
from charm4py import _ckdyn
import numpy as np

try:
    from numba import cuda
except ImportError:
    cuda = None

N = 1 << 20


def as_host(d):
    """ a numba device array, or anything with __cuda_array_interface__, to numpy """
    if not hasattr(d, 'copy_to_host'):
        d = cuda.as_cuda_array(d)
    return d.copy_to_host()


class A(Chare):

    def __init__(self):
        self.idx = self.thisIndex[0] if isinstance(self.thisIndex, tuple) else self.thisIndex
        self.partner = self.thisProxy[1 - self.idx] if self.idx < 2 else None
        self.posted = None

    def post(self, method, slot):
        self.posted = cuda.device_array(N, dtype=np.int8)
        self.post_buffer(getattr(self, method), slot, self.posted)

    def recv(self, a, b, s, h, d, posted, done):
        """ element 0 """
        assert (a, b, s) == (1, 2, 'hello'), (a, b, s)
        assert np.array_equal(h, np.ones(N, dtype=np.int8)), 'host argument differs'
        if posted:
            assert d is self.posted, 'device argument: got a %s, not the posted array' % type(d).__name__
        else:
            assert isinstance(d, _ckdyn.DeviceBuffer), type(d)
            assert d.nbytes == N and d.shape == (N,) and d.typestr == '|i1', (d.nbytes, d.shape, d.typestr)
            d = cuda.as_cuda_array(d)
        assert (as_host(d) == 5).all(), 'device argument differs'
        self.partner.back(2, 3, d, posted, done)

    def back(self, a, b, d, posted, done):
        """ element 1 """
        assert (a, b) == (2, 3)
        if posted:
            assert d is self.posted, 'returned device argument is not the posted array'
        else:
            assert isinstance(d, _ckdyn.DeviceBuffer), type(d)
        done.send(bool((as_host(d) == 5).all()))

    @coro
    def run(self, posted):
        """ element 1 """
        if posted:
            self.partner.post('recv', 4, ret=True).get()
            self.post('back', 2)
        d = cuda.to_device(np.full(N, 5, dtype=np.int8))
        done = Future()
        self.partner.recv(1, 2, 'hello', np.ones(N, dtype=np.int8), d, posted, done)
        return done.get()

    @coro
    def channel_run(self):
        """ both elements: one message each way over a channel """
        ch = Channel(self, self.partner)
        if self.idx == 1:
            ch.send(1, 2, 'hello', np.ones(N, dtype=np.int8), cuda.to_device(np.full(N, 5, dtype=np.int8)))
            d = ch.recv()
            assert isinstance(d, _ckdyn.DeviceBuffer), type(d)
            return bool((as_host(d) == 5).all())
        a, b, s, h, d = ch.recv()
        assert (a, b, s) == (1, 2, 'hello') and np.array_equal(h, np.ones(N, dtype=np.int8))
        assert isinstance(d, _ckdyn.DeviceBuffer), type(d)
        assert (as_host(d) == 5).all(), 'channel: device argument differs'
        ch.send(d)   # a DeviceBuffer is itself a device array: sent device to device
        return True


def main(args):
    if cuda is None or not cuda.is_available() or not _ckdyn.has_device_support():
        why = ('numba is not installed' if cuda is None else 'no GPU is visible to numba'
               if not cuda.is_available() else 'this Charm++ has no CUDA or HIP support')
        print('SKIP single_array.py: ' + why)
        charm.exit()
        return
    if charm.numPes() < 2:
        print('SKIP single_array.py: needs 2 PEs')
        charm.exit()
        return
    for kind, chares in (('array', Array(A, 2)), ('group', Group(A))):
        charm.awaitCreation(chares)
        for posted in (False, True):
            assert chares[1].run(posted, ret=True).get() is True, '%s posted=%s: round trip differs' % (kind, posted)
        f = [chares[i].channel_run(ret=True) for i in range(2)]
        assert all(x.get() is True for x in f), kind + ': channel exchange failed'
        print('single_array: %s passed (DeviceBuffer, posted receive, channel)' % kind)
    charm.exit()


charm.start(main)

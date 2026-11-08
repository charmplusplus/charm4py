"""Device arrays as arguments (SPEC-3.md section 5). Needs a CUDA build of Charm++
(_ckdyn.has_device_support()) and cupy; prints SKIP and exits otherwise.

Run: python -m charmrun.start tests/thin/device.py +p2

- a cupy array sent to an element on another PE arrives as a _ckdyn.DeviceBuffer;
  cupy.asarray of it compares equal to what was sent;
- the receiver sends it back (a cupy array over the received buffer): a round trip;
- charm.hapiAddCudaCallback on a cupy stream after a kernel completes a future with
  None, and the kernel's result is complete when it does.
"""
from charm4py import charm, Chare, Array, Future
from charm4py import _ckdyn

try:
    import cupy
except ImportError:
    cupy = None

N = 2 * 1024 * 1024 // 8   # 2 MB of float64


def pattern(seed):
    return cupy.arange(N, dtype=cupy.float64) * 0.5 + seed


class D(Chare):

    def recv(self, buf, seed, done):
        assert isinstance(buf, _ckdyn.DeviceBuffer), type(buf)
        assert buf.nbytes == N * 8 and buf.shape == (N,) and buf.typestr == '<f8'
        a = cupy.asarray(buf)   # no copy: a view of the received device memory
        assert a.data.ptr == buf.ptr
        assert bool(cupy.all(a == pattern(seed))), 'recv: data differs'
        self.kept = a
        self.thisProxy[0].back(a, seed, done)

    def back(self, buf, seed, done):
        assert isinstance(buf, _ckdyn.DeviceBuffer), type(buf)
        a = cupy.asarray(buf)
        assert bool(cupy.all(a == pattern(seed))), 'back: data differs'
        done.send(True)

    def check_kept(self, seed):
        return bool(cupy.all(self.kept == pattern(seed)))


def main(args):
    if cupy is None or not _ckdyn.has_device_support():
        print('SKIP device.py: %s' % ('cupy is not installed' if cupy is None
                                      else 'this Charm++ has no CUDA or HIP support'))
        charm.exit()
        return
    arr = Array(D, 2)
    charm.awaitCreation(arr)

    x = pattern(5.0)
    cupy.cuda.Device().synchronize()
    done = Future()
    arr[1].recv(x, 5.0, done)
    assert done.get() is True
    assert arr[1].check_kept(5.0, ret=True).get() is True, 'kept device array lost its data'

    # a kernel on a stream, then a HAPI callback on that stream
    s = cupy.cuda.Stream(non_blocking=True)
    with s:
        y = cupy.arange(N, dtype=cupy.float64) * 3.0
    f = Future()
    charm.hapiAddCudaCallback(s.ptr, f)
    assert f.get() is None
    assert bool(cupy.all(y == cupy.arange(N, dtype=cupy.float64) * 3.0))
    print('device: round trip of a %d MB cupy array and a HAPI callback passed' % (N * 8 // (1 << 20)))
    charm.exit()


charm.start(main)

"""charm.exit() after a compiled CuPy kernel: no CUDADriverError at process exit.
Needs a CUDA build of Charm++ (_ckdyn.has_device_support()) and cupy; prints SKIP
and exits otherwise.

Run: python -m charmrun.start tests/thin/cupy_exit.py +p2

charm.exit() ends in ConverseExit, which calls exit(); the binding's C atexit
handler then runs Python's atexit handlers, and CuPy's unload the modules it
compiled. When that handler ran after CUDA's own teardown at exit, every process
that had compiled a kernel printed "CUDADriverError" from Module.__dealloc__
("Exception ignored in: 'cupy.cuda.function.Module.__dealloc__'"). Check the
output of this test for those lines: the exit code alone does not show them.
"""
from charm4py import charm, Chare, Group
from charm4py import _ckdyn

try:
    import cupy
except ImportError:
    cupy = None


class K(Chare):

    def run(self):
        # an elementwise kernel defined here is compiled (and loaded as a module)
        # in this process, not taken from CuPy's prebuilt kernels
        axpb = cupy.ElementwiseKernel('float64 x, float64 a, float64 b', 'float64 y',
                                      'y = a * x + b', 'cupy_exit_axpb')
        x = cupy.arange(1024, dtype=cupy.float64)
        y = axpb(x, 2.0, 1.0)
        return bool(cupy.all(y == 2.0 * x + 1.0))


def main(args):
    if cupy is None or not _ckdyn.has_device_support():
        print('SKIP cupy_exit.py: %s' % ('cupy is not installed' if cupy is None
                                         else 'this Charm++ has no CUDA or HIP support'))
        charm.exit()
        return
    try:
        ndev = cupy.cuda.runtime.getDeviceCount()
    except Exception:
        ndev = 0
    if ndev == 0:
        print('SKIP cupy_exit.py: no GPU')
        charm.exit()
        return
    g = Group(K)
    fs = [g[pe].run(ret=True) for pe in range(charm.numPes())]   # every PE compiles it
    assert all(f.get() for f in fs), 'kernel result differs'
    print('cupy_exit: kernel ran on %d PEs; exiting (no driver error may follow)' % charm.numPes())
    charm.exit()


charm.start(main)

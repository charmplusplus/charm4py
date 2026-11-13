"""charm.exit() after a compiled CuPy kernel: no CUDADriverError at process exit.
Needs a CUDA build of Charm++ (_ckdyn.has_device_support()) and cupy; prints SKIP
and exits otherwise.

Run: python -m charmrun.start tests/thin/cupy_exit.py +p2

charm.exit() ends in ConverseExit, which calls exit(); the binding's C atexit
handler then runs Python's atexit handlers, and CuPy's unload the modules it
compiled. When that handler ran after CUDA's own teardown at exit, every process
that had compiled a kernel printed "CUDADriverError" from Module.__dealloc__
("Exception ignored in: 'cupy.cuda.function.Module.__dealloc__'") and still
exited with status 0.

The test checks this itself: a sys.unraisablehook records every exception that
Python reports as ignored (Cython __dealloc__ errors go through it), and an atexit
function registered BEFORE cupy is imported ends the process with os._exit(1) when
anything was recorded. Python's atexit runs handlers last-registered first, so
this function runs after CuPy's.
"""
import atexit
import os
import sys

_ignored = []   # text of every exception reported through sys.unraisablehook


def _record_unraisable(u):
    # keep only text: holding u.object or the exception could resurrect objects
    _ignored.append('%s: %s (%s)' % (getattr(u.exc_type, '__name__', u.exc_type),
                                     u.exc_value, u.err_msg or 'Exception ignored'))
    sys.__unraisablehook__(u)   # print it as Python would


def _fail_if_ignored():
    if _ignored:
        print('FAIL cupy_exit.py: pid %d: %d exception(s) ignored, first: %s'
              % (os.getpid(), len(_ignored), _ignored[0]), file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


sys.unraisablehook = _record_unraisable
atexit.register(_fail_if_ignored)   # before `import cupy`: runs after CuPy's handlers

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

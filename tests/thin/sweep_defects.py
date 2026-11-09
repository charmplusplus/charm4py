"""Defects found by the step 7 example sweep (logs/step-7/examples-sweep.md) and the
migration guide, one phase each.

Run: python -m charmrun.start tests/thin/sweep_defects.py +p1   (and +p2, +p4)

Phases:
- reduce1: a reduction with one contributor gives the same result type as one with
  several. A built-in reducer on an object with __array__ (as a torch tensor) or on
  a list of mixed numbers returns a numpy array for 1 and for 3 contributors; numpy
  contributions give numpy arrays; gather gives a list.
- zcprotect: a numpy array sent as an NCPY slot (2 MB, threshold lowered to 64 KB)
  is read-only from the send until the transfer is done, and writeable again after;
  a write in between raises ValueError instead of changing what the receiver gets;
  an array that was read-only before the send stays read-only.
- rayshare (two PEs or more; the pool needs a worker PE): one ray.put reference
  passed to 12 tasks. Every copy of the reference that a PE forwards is a separate
  object, and each must be counted on its own when its copies are deleted (the
  borrowed-futures table was keyed by (store_id, depth), so later copies replaced
  earlier ones and the deletion notices raised KeyError).
- atexit: with arguments `atexit DIR`, every process registers a Python atexit
  handler that writes DIR/atexit.<pe>; the program ends with charm.exit(). Check
  afterwards with `python tests/thin/sweep_defects.py check-atexit DIR NPES` (not
  part of the auto_test run, which cannot look at files after the processes end).
"""
import atexit
import gc
import os
import sys

if len(sys.argv) >= 2 and sys.argv[1] == 'check-atexit':
    d, n = sys.argv[2], int(sys.argv[3])
    missing = [pe for pe in range(n) if not os.path.isfile(os.path.join(d, 'atexit.%d' % pe))]
    if missing:
        print('atexit: no file from PE(s)', missing)
        sys.exit(1)
    print('atexit: PASS, handlers ran on all %d PEs' % n)
    sys.exit(0)

from charm4py import charm, Chare, Array, Future, Reducer, coro, ray
import numpy as np

ATEXIT_DIR = sys.argv[2] if len(sys.argv) >= 3 and sys.argv[1] == 'atexit' else None


def _write_atexit_file():
    with open(os.path.join(ATEXIT_DIR, 'atexit.%d' % charm.myPe()), 'w') as f:
        f.write('ran\n')


if ATEXIT_DIR is not None:
    atexit.register(_write_atexit_file)


class ArrayLike(object):
    """ not a numpy array, converts to one (as a torch tensor does) """
    def __init__(self, a):
        self.a = a

    def __array__(self, dtype=None, copy=None):
        return self.a if dtype is None else self.a.astype(dtype)


class R(Chare):

    @coro
    def run(self, n, done):
        i = self.thisIndex[0]
        r = self.allreduce(ArrayLike(np.ones(5) * (i + 1)), Reducer.sum).get()
        assert isinstance(r, np.ndarray), 'n=%d: array-like sum gave %s' % (n, type(r).__name__)
        assert np.array_equal(r, np.ones(5) * (n * (n + 1) // 2)), r
        r = self.allreduce([1, 2.5], Reducer.max).get()
        assert isinstance(r, np.ndarray) and list(r) == [1.0, 2.5], 'n=%d: mixed list max gave %r' % (n, r)
        r = self.allreduce(np.arange(3, dtype=np.int64) + i, Reducer.sum).get()
        assert isinstance(r, np.ndarray) and r.dtype == np.int64, 'n=%d: ndarray sum gave %r' % (n, r)
        r = self.allreduce(i, Reducer.gather).get()
        assert r == list(range(n)), 'n=%d: gather gave %r' % (n, r)
        self.reduce(done)


class Z(Chare):

    def recv(self, x, expect, reply):
        reply.send(bool(np.array_equal(x, expect)))


def phase_reduce1():
    for n in (1, 3):
        a = Array(R, n)
        done = Future()
        a.run(n, done)
        done.get()
    print('reduce1: PASS')


def wait_until(cond, what):
    for _ in range(500):
        if cond():
            return
        charm.sleep(0.01)
    raise AssertionError(what)


def phase_zcprotect():
    old = charm.options.zerocopy_threshold
    charm.options.zerocopy_threshold = 64 * 1024
    z = Array(Z, charm.numPes())
    charm.awaitCreation(z)
    zz = z[charm.numPes() - 1]   # on another PE when there are several
    n = 2 * 1024 * 1024 // 8
    expect = np.arange(n, dtype=np.float64)

    a = expect.copy()
    assert a.flags.writeable
    reply = Future()
    zz.recv(a, expect, reply)
    assert not a.flags.writeable, 'the sent array is writeable while its transfer is pending'
    try:
        a[0] = -1.0
        raise AssertionError('a write to the sent array did not raise')
    except ValueError:
        pass
    assert reply.get() is True, 'the receiver got different contents'
    wait_until(lambda: a.flags.writeable, 'the sent array is still read-only after its transfer')
    a[0] = -1.0   # writeable again

    # the same array in two pending sends: read-only until both are done
    b = expect.copy()
    r1, r2 = Future(), Future()
    zz.recv(b, expect, r1)
    zz.recv(b, expect, r2)
    assert not b.flags.writeable
    assert r1.get() is True and r2.get() is True
    wait_until(lambda: b.flags.writeable, 'an array sent twice stays read-only')

    # read-only before the send: stays read-only
    c = expect.copy()
    c.flags.writeable = False
    r3 = Future()
    zz.recv(c, expect, r3)
    assert r3.get() is True
    from charm4py import _ckdyn
    wait_until(lambda: _ckdyn.zerocopy_counts()[0] == _ckdyn.zerocopy_counts()[1],
               'NCPY slots not done')
    assert not c.flags.writeable, 'a read-only array became writeable after its send'
    charm.options.zerocopy_threshold = old
    print('zcprotect: PASS')


def model(x):
    return x * 2


@ray.remote
def task(m, i):
    return m(i)


def phase_rayshare():
    if charm.numPes() < 2:
        print('rayshare: skipped on one PE (the pool needs a worker PE)')
        return
    ray.init()
    for n in (4, 12):
        ref = ray.put(model)
        refs = [task.remote(ref, i) for i in range(n)]
        assert ray.get(refs) == [2 * i for i in range(n)]
        del ref, refs
        gc.collect()
    # the deletion notices of the forwarded copies arrive here; a miscount aborts
    charm.waitQD()
    print('rayshare: PASS')


def main(args):
    phase_reduce1()
    phase_zcprotect()
    phase_rayshare()
    if ATEXIT_DIR is not None:
        print('atexit: handlers registered on every PE, files in', ATEXIT_DIR)
    print('sweep_defects: all phases passed')
    charm.exit()


charm.start(main)

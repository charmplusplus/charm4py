"""Numeric reductions keep the contributors' array shape (issue #321).

Run: python tests/thin/reduction_shape.py +pe 1   (or python -m charmrun.start +p2 ...)

A contribution of an n-dimensional numpy array (n >= 2) carries its shape
(CkDynContributeShaped), and the built-in reducer's result is an array of that
shape, for a whole-array reduction and for a section reduction alike.

1. (2, 3) float64 arrays, Reducer.sum over the whole array and over a section.
2. (2, 3) int32 arrays, Reducer.max over the whole array.
3. (2, 2, 2) float64 arrays over the whole array, and over a section with the
   result delivered to a SECTION callback (a method of the section members).
4. A (1, 1) array stays an array; a 1-D array of length 1 is a scalar, as
   before.
5. Contributions of two shapes to one reduction abort the program with a
   CkDyn message naming both shapes: checked in a child process
   (``--mixed``), which this test starts with one PE.
"""
import os
import subprocess
import sys

import numpy as np
from charm4py import charm, Chare, Array, Reducer, Future

N = 4


class W(Chare):

    def base(self, shape, dtype):
        n = int(np.prod(shape))
        return np.arange(1, n + 1, dtype=dtype).reshape(shape)

    def go(self, cb, sec, shape, dtype, reducer):
        a = self.base(shape, dtype) * (self.thisIndex[0] + 1)
        if sec is not None:
            self.contribute(a, reducer, cb, sec)
        else:
            self.contribute(a, reducer, cb)

    def to_section(self, sec, f):
        # a section reduction whose result goes to a method of the section members
        self.f = f
        self.sec = sec
        a = self.base((2, 2, 2), np.float64) * (self.thisIndex[0] + 1)
        self.contribute(a, Reducer.sum, sec.got, sec)

    def got(self, r):
        # every member checks the shaped result and reports back
        assert isinstance(r, np.ndarray) and r.shape == (2, 2, 2), (type(r), getattr(r, 'shape', None))
        expect = self.base((2, 2, 2), np.float64) * (2 + 3)
        assert np.array_equal(r, expect), r
        self.contribute(1, Reducer.sum, self.f, self.sec)

    def mixed(self, f):
        shape = (2, 3) if self.thisIndex[0] % 2 == 0 else (3, 2)
        self.contribute(np.ones(shape), Reducer.sum, f)


def reduce(arr, shape, dtype, reducer, sec=None):
    f = Future()
    (sec if sec is not None else arr).go(f, sec, shape, dtype, reducer)
    return f.get()


def check(what, r, expect):
    assert isinstance(r, np.ndarray), (what, type(r))
    assert r.shape == expect.shape, (what, r.shape, expect.shape)
    assert r.dtype == expect.dtype, (what, r.dtype, expect.dtype)
    assert np.array_equal(r, expect), (what, r, expect)
    print('  %s: shape %s ok' % (what, r.shape))


def base(shape, dtype):
    return np.arange(1, int(np.prod(shape)) + 1, dtype=dtype).reshape(shape)


def run_mixed():
    """ the child process: two shapes in one reduction, which must abort """
    def main(args):
        arr = Array(W, N)
        f = Future()
        arr.mixed(f)
        print('MIXED-SHAPE RESULT', f.get())
        charm.exit()
    charm.start(main)


def check_mixed():
    env = {k: v for k, v in os.environ.items() if not k.startswith(('LCT_', 'LCI_', 'PMI_', 'PMIX_'))}
    p = subprocess.run([sys.executable, os.path.abspath(__file__), '--mixed', '+pe', '1'], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    out = p.stdout
    assert p.returncode != 0, 'contributions of two shapes did not abort:\n' + out
    assert 'MIXED-SHAPE RESULT' not in out, out
    assert ('contributions to one reduction have different shapes' in out and '(2, 3)' in out
            and '(3, 2)' in out), out
    print('  mixed shapes: the child aborted (exit status %d) with the CkDyn message' % p.returncode)


def main(args):
    arr = Array(W, N)
    total = sum(range(1, N + 1))   # sum of (index + 1)

    # 1. (2, 3) float64, whole array and section
    check('whole array sum (2, 3)', reduce(arr, (2, 3), np.float64, Reducer.sum), base((2, 3), np.float64) * total)
    sec = arr[1:3]
    check('section sum (2, 3)', reduce(arr, (2, 3), np.float64, Reducer.sum, sec), base((2, 3), np.float64) * (2 + 3))

    # 2. (2, 3) int32 max
    check('whole array max (2, 3) int32', reduce(arr, (2, 3), np.int32, Reducer.max), base((2, 3), np.int32) * N)

    # 3. (2, 2, 2): whole array, and a section reduction to a SECTION callback
    check('whole array sum (2, 2, 2)', reduce(arr, (2, 2, 2), np.float64, Reducer.sum),
          base((2, 2, 2), np.float64) * total)
    f = Future()
    sec.to_section(sec, f)
    assert f.get() == 2, 'section members that checked the (2, 2, 2) result'
    print('  section sum (2, 2, 2) to a SECTION callback: both members got shape (2, 2, 2)')

    # 4. one element: (1, 1) stays an array, a 1-D length-1 array is a scalar
    check('whole array sum (1, 1)', reduce(arr, (1, 1), np.float64, Reducer.sum), np.array([[float(total)]]))
    r = reduce(arr, (1,), np.float64, Reducer.sum)
    assert isinstance(r, float) and r == total, r
    print('  whole array sum (1,): scalar %s ok' % r)

    # 5. mixed shapes abort (one child process; only from the first process)
    check_mixed()
    print('reduction_shape: all checks passed')
    charm.exit()


if __name__ == '__main__':
    if '--mixed' in sys.argv:
        run_mixed()
    else:
        charm.start(main)

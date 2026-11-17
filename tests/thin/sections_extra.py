"""Section checks of step 4b that the 1.x tests do not make (SPEC-4b.md 2-3).

Run: python tests/thin/sections_extra.py +pe 1   (or lcrun -n 2 ... +pe 2)

1. A 0-dimensional numpy array sent to a section arrives as a 0-dimensional
   array (the binding tags it TAG_ND0), not as a reduction result.
2. A built-in reduction over a section whose callback is a method of the section
   proxy (a SECTION callback) delivers the reduced numpy array to every member.
3. charm.combine with a group raises TypeError (DESIGN.md 5.4 item 15).
"""
from charm4py import charm, Chare, Array, Group, Reducer, Future
import numpy as np


class T(Chare):

    def setsec(self, s):
        self.sec = s

    def recv(self, x, f):
        self.contribute((type(x).__name__, getattr(x, 'ndim', None)), Reducer.gather, f, self.sec)

    def go(self, f):
        self.f = f
        self.contribute(np.arange(3.0), Reducer.sum, self.sec.result, self.sec)

    def result(self, v):
        assert isinstance(v, np.ndarray) and list(v) == [0.0, 4.0, 8.0], v
        self.contribute(None, None, self.f, self.sec)


def main(args):
    a = Array(T, 8)
    s = a[0:8:2]
    a.setsec(s, awaitable=True).get()
    f = Future()
    s.recv(np.array(3.0), f)
    assert f.get() == [('ndarray', 0)] * 4
    f = Future()
    s.go(f)
    f.get()
    try:
        charm.combine(s, Group(T))
        raise AssertionError('charm.combine with a group did not raise TypeError')
    except TypeError:
        pass
    print('sections_extra: PASS')
    exit()


charm.start(main)

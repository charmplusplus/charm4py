"""Reducer.sum and Reducer.product of booleans count, as np.sum and np.prod do.

Run: python tests/thin/reduction_bool.py +pe 1   (or lcrun -n N ... +pe N)

Booleans have no Charm++ sum or product reducer. The result must be the
widened integer result (np.sum of bools is an int), not the logical OR/AND
that a bool-typed numpy add/multiply gives. A scalar bool contribution gives
a scalar number; a bool array gives an integer array.
"""
from charm4py import charm, Chare, Group, Array, Reducer, Future
import numpy as np


class G(Chare):

    def run(self, kind, f):
        me = charm.myPe()
        if kind == 'sum_scalar':
            self.reduce(f, True, Reducer.sum)
        elif kind == 'sum_npscalar':
            self.reduce(f, np.bool_(True), Reducer.sum)
        elif kind == 'sum_list':
            self.reduce(f, [True, me == 0], Reducer.sum)
        elif kind == 'sum_array':
            self.reduce(f, np.array([True, me == 0]), Reducer.sum)
        elif kind == 'prod_scalar':
            self.reduce(f, True, Reducer.product)
        elif kind == 'prod_array':
            self.reduce(f, np.array([True, me == 0]), Reducer.product)
        elif kind == 'or_array':
            self.reduce(f, np.array([me == 0, False]), Reducer.logical_or)


def check(name, got, expected):
    ok = np.array_equal(np.asarray(got), np.asarray(expected))
    if isinstance(expected, (int, np.integer)):
        ok = ok and np.ndim(got) == 0 and not isinstance(got, (bool, np.bool_))
    else:
        ok = ok and np.asarray(got).dtype.kind in 'iu'
    print(name, repr(got), 'expected', repr(expected))
    assert ok, name + ': got ' + repr(got) + ', expected ' + repr(expected)


def main(args):
    n = charm.numPes()
    g = Group(G)
    cases = [('sum_scalar', n), ('sum_npscalar', n), ('sum_list', [n, 1]),
             ('sum_array', [n, 1]), ('prod_scalar', 1),
             ('prod_array', [1, 1 if n == 1 else 0])]
    for kind, expected in cases:
        f = Future()
        g.run(kind, f)
        check(kind, f.get(), expected)
    f = Future()
    g.run('or_array', f)
    r = f.get()
    print('or_array', repr(r))
    assert np.array_equal(r, [True, False]) and np.asarray(r).dtype == bool
    print('reduction_bool: all checks passed')
    charm.exit()


charm.start(main)

"""Reductions whose result goes to a section, and Python reducer results.

Run: python tests/thin/section_reduce.py +pe 1   (or lcrun -n N ... +pe N, N = 2, 4)

An array of 8 elements and a group. The members of the array section (even
indices) and of the group section (even PEs) check each result they receive
and answer with a section reduction, so that main checks the member count.

1. Every element reduces with Reducer.sum to a method of the array section
   (CkDynContribute with a SECTION callback, which reaches members on every PE).
2. Every group branch reduces to a method of the group section; every array
   element reduces to the group section.
3. Section allreduce over the array section and over the group section (a
   section reduction to a SECTION callback with a non-zero reference number).
4. A custom Python reducer to the array section, from the whole array and as a
   section reduction.
5. numpy reductions to the array section: float64, int8 with negative values,
   uint8 (the *_char reducers are I8 or U8 by the sign of plain char).
6. Reducer.gather to the array section.
7. A Python reducer that returns a freshly built numpy array: three rounds in
   which every element contributes the previous round's result (the reduce
   hook's result builder copies the array before the reducer's temporary is
   freed).
8. int8 sum with no built-in I8 reducer in the table (the table of aarch64 and
   POWER Linux, where plain char is unsigned): the Python reducer combines it.
"""
from charm4py import charm, Chare, Array, Group, Reducer, Future, coro
from charm4py.reduction import SUM, ELEM_I8
import numpy as np

N = 8
BIG = 50000


def pair_reducer(contribs):
    return [sum(c[0] for c in contribs), max(c[1] for c in contribs)]


def fresh_np_reducer(contribs):
    # a new array on every call; the contributions are not modified
    out = np.empty(len(contribs[0]), dtype=np.float64)
    out[:] = 0.0
    for c in contribs:
        out += c
    return out * 1.0


Reducer.addReducer(pair_reducer)
Reducer.addReducer(fresh_np_reducer)


class E(Chare):

    def setup(self, asec, gsec):
        self.asec = asec
        self.gsec = gsec
        self.idx = self.thisIndex[0]

    def expect(self, f, expected):
        self.f = f
        self.expected = expected

    def answer(self, ok):
        assert ok, (self.idx, 'unexpected value')
        self.contribute(1, Reducer.sum, self.f, self.asec)

    # 1 and 2
    def go_sum(self, to_group):
        target = self.gsec.got if to_group else self.asec.got
        self.reduce(target, self.idx + 1, Reducer.sum)

    def got(self, v):
        self.answer(v == self.expected)

    # 3
    @coro
    def go_allreduce(self, f):
        v = self.allreduce(self.idx, Reducer.sum, section=self.asec).get()
        assert v == sum(range(0, N, 2)), v
        v = self.allreduce(np.array([self.idx, 1]), Reducer.sum, section=self.asec).get()
        assert list(v) == [sum(range(0, N, 2)), N // 2], v
        self.contribute(1, Reducer.sum, f, self.asec)

    # 4
    def go_custom(self):
        self.reduce(self.asec.got_list, [self.idx, self.idx], Reducer.pair_reducer)

    def go_custom_sec(self):
        self.contribute([self.idx, self.idx], Reducer.pair_reducer, self.asec.got_list, self.asec)

    def got_list(self, v):
        self.answer(list(v) == self.expected)

    # 5
    def go_np(self, kind):
        if kind == 'f8':
            x = np.arange(4, dtype=np.float64) * self.idx
        elif kind == 'i1':
            x = np.full(3, self.idx - 4, dtype=np.int8)
        else:
            x = np.full(3, self.idx * 3, dtype=np.uint8)
        self.reduce(self.asec.got_np, x, Reducer.sum)

    def got_np(self, v):
        exp = np.array(self.expected[0], dtype=self.expected[1])
        self.answer(isinstance(v, np.ndarray) and v.dtype == exp.dtype and np.array_equal(v, exp))

    # 6
    def go_gather(self):
        self.reduce(self.asec.got_list, self.idx, Reducer.gather)

    # 7
    def go_fresh(self, prev, rnd, f):
        if rnd == 0:
            x = np.arange(BIG, dtype=np.float64) + self.idx
        else:
            x = prev
        self.f7 = f
        self.reduce(self.thisProxy.fresh_result, x, Reducer.fresh_np_reducer)

    def fresh_result(self, v):
        self.contribute(v, Reducer.fresh_np_reducer, self.f7)

    # 8
    def go_i8(self):
        self.reduce(self.asec.got_np, np.full(3, self.idx - 4, dtype=np.int8), Reducer.sum)


class G(Chare):

    def setup(self, gsec):
        self.gsec = gsec

    def expect(self, f, expected):
        self.f = f
        self.expected = expected

    def go_sum(self):
        self.reduce(self.gsec.got, charm.myPe() + 1, Reducer.sum)

    def got(self, v):
        assert v == self.expected, (charm.myPe(), v, self.expected)
        self.contribute(1, Reducer.sum, self.f, self.gsec)

    @coro
    def go_allreduce(self, f):
        v = self.allreduce(charm.myPe(), Reducer.sum, section=self.gsec).get()
        assert v == sum(range(0, charm.numPes(), 2)), v
        self.contribute(1, Reducer.sum, f, self.gsec)

    def drop_i8(self):
        charm.redMgr.red_table[SUM][ELEM_I8] = None


def main(args):
    npes = charm.numPes()
    a = Array(E, N)
    g = Group(G)
    asec = a[0:N:2]
    gsec = g[::2]
    nasec, ngsec = N // 2, len(range(0, npes, 2))
    a.setup(asec, gsec, awaitable=True).get()
    g.setup(gsec, awaitable=True).get()

    def phase(name, expected, start, n=nasec, coll=None):
        f = Future()
        (coll or asec).expect(f, expected, awaitable=True).get()
        start()
        got = f.get()
        assert got == n, (name, got, n)
        print('section_reduce: %s ok' % name)

    # 1: array -> array section
    phase('array sum to array section', N * (N + 1) // 2, lambda: a.go_sum(False))
    # 2: group -> group section, array -> group section
    phase('group sum to group section', npes * (npes + 1) // 2, lambda: g.go_sum(), ngsec, gsec)
    phase('array sum to group section', N * (N + 1) // 2, lambda: a.go_sum(True), ngsec, gsec)
    # 3: section allreduce (reference number on a SECTION callback)
    f1, f2 = Future(), Future()
    asec.go_allreduce(f1)
    gsec.go_allreduce(f2)
    assert f1.get() == nasec and f2.get() == ngsec
    print('section_reduce: section allreduce ok')
    # 4: custom Python reducer to a section
    phase('custom reducer to array section', [N * (N - 1) // 2, N - 1], lambda: a.go_custom())
    phase('custom section reduction to array section', [sum(range(0, N, 2)), N - 2], lambda: asec.go_custom_sec())
    # 5: numpy reductions
    phase('numpy float64 sum', ([0.0, 28.0, 56.0, 84.0], 'float64'), lambda: a.go_np('f8'))
    phase('numpy int8 sum', ([-4] * 3, 'int8'), lambda: a.go_np('i1'))
    phase('numpy uint8 sum', ([84] * 3, 'uint8'), lambda: a.go_np('u1'))
    # 6: gather
    phase('gather to array section', list(range(N)), lambda: a.go_gather())
    # 7: a freshly built numpy result, three rounds
    # round 0 reduces arange + idx to r0; each round the result is broadcast and
    # every element contributes it to f (factor N), and the next round starts
    # from f's value (factor N again)
    prev = None
    r0 = np.arange(BIG, dtype=np.float64) * N + N * (N - 1) // 2
    for rnd in range(3):
        f = Future()
        a.go_fresh(prev, rnd, f)
        prev = f.get()
        assert isinstance(prev, np.ndarray) and prev.dtype == np.float64, type(prev)
        assert np.array_equal(prev, r0 * float(N ** (2 * rnd + 1))), (rnd, prev[:4])
    print('section_reduce: fresh numpy result over 3 rounds ok')
    # 8: int8 without a built-in I8 reducer
    g.drop_i8(awaitable=True).get()
    phase('int8 sum through the Python reducer', ([-4] * 3, 'int8'), lambda: a.go_i8())
    print('section_reduce: PASS')
    exit()


charm.start(main)

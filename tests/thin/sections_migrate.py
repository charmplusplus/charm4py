"""Section members that migrate (DESIGN.md 5.4 item 10, SPEC-4c.md 4).

Run: lcrun -n 2 python tests/thin/sections_migrate.py +pe 2 +balancer RotateLB
     (any process count >= 2)

An array is split into two sections (even and odd indices) with charm.split, and the
two are combined with charm.combine, so every element is a member of two sections.
Each round: a multicast to each section, a sum reduction over each section and over
the combined section, then AtSync. RotateLB moves every element every period, so each
round after the first runs with every member on a different PE than in the round
before (CkDyn pups the section cookies; CkMulticast rebuilds its tree).
"""
from charm4py import charm, Chare, Array, Reducer, Future
import numpy as np

ROUNDS = 5


class T(Chare):

    def __init__(self):
        self.pes = [charm.myPe()]

    def setsecs(self, secs, comb):
        self.sec = secs[self.thisIndex[0] % 2]
        self.comb = comb

    def ping(self, r, f):
        # a multicast to the member's section: reply through a section reduction
        self.contribute(self.thisIndex[0] * 1000 + r, Reducer.sum, f, self.sec)

    def round(self, r, fsec, fcomb, fsync):
        v = np.array([self.thisIndex[0], r], dtype='int64')
        self.contribute(v, Reducer.sum, fsec[self.thisIndex[0] % 2], self.sec)
        self.contribute(v, Reducer.sum, fcomb, self.comb)
        self.fsync = fsync
        self.AtSync()

    def resumeFromSync(self):
        self.pes.append(charm.myPe())
        self.contribute((self.thisIndex[0], self.pes), Reducer.gather, self.fsync)


def main(args):
    npes = charm.numPes()
    assert npes >= 2, 'run with at least 2 PEs (one per process)'
    n = 4 * npes
    a = Array(T, n, useAtSync=True)
    even = list(range(0, n, 2))
    odd = list(range(1, n, 2))
    secs = charm.split(a, 2, elems=[[(i,) for i in even], [(i,) for i in odd]])
    comb = charm.combine(secs[0], secs[1])
    a.setsecs(secs, comb, awaitable=True).get()
    members = [even, odd]
    for r in range(ROUNDS):
        for k in range(2):
            f = Future()
            secs[k].ping(r, f)
            assert f.get() == sum(i * 1000 + r for i in members[k]), ('ping', r, k)
        fsec = [Future(), Future()]
        fcomb = Future()
        fsync = Future()
        a.round(r, fsec, fcomb, fsync)
        for k in range(2):
            assert list(fsec[k].get()) == [sum(members[k]), r * len(members[k])], ('section', r, k)
        assert list(fcomb.get()) == [sum(range(n)), r * n], ('combined', r)
        pes = dict(fsync.get())
        assert all(p[-1] != p[-2] for p in pes.values()), ('an element did not move', r, pes)
    print('sections_migrate: %d elements, %d rounds; element 0 PEs %s' % (n, ROUNDS, pes[0]))
    print('sections_migrate: PASS')
    exit()


charm.start(main)

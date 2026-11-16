"""
A broadcast handler destroys another local element of the same array.

Run: python tests/thin/destroy_sibling.py +pe 1   (and lcrun -n 2 ... +pe 2)

The broadcast go() reaches every local element through one deliver_many call.
The first element of each local pair to run calls an inline entry die() on its
partner, which calls self.ckDestroy(): the partner is deleted while the
broadcast is still being delivered on this PE. The partner must then be skipped
(before the fix the loop used its stale handle and the run aborted with
AttributeError: 'E' object has no attribute '_ckdyn_target').

Checks: in every pair exactly one element ran go(), the killed elements are gone
from charm.arrays, and the survivors still answer.
"""

from charm4py import charm, Chare, Group, Array, coro, entry

LOCAL = {}     # index -> element, the live elements of this process
RAN = []       # indices whose go() ran in this process
KILLED = []    # indices destroyed by a sibling in this process


def check(cond, what):
    if not cond:
        charm.abort('FAIL ' + what)
    print('[PE %d] PASS %s' % (charm.myPe(), what))


class E(Chare):

    def __init__(self):
        LOCAL[self.thisIndex[0]] = self

    def go(self):
        i = self.thisIndex[0]
        RAN.append(i)
        p = i ^ 1
        if p in LOCAL:
            self.thisProxy[p].die()

    @entry(inline=True)
    def die(self):
        i = self.thisIndex[0]
        del LOCAL[i]
        KILLED.append(i)
        self.ckDestroy()

    def val(self):
        return self.thisIndex[0]


class Probe(Chare):

    def state(self, aid):
        return (sorted(RAN), sorted(KILLED), sorted(i[0] for i in charm.arrays.get(aid, {})))


class Main(Chare):

    def __init__(self, args):
        self.thisProxy.run()

    @coro
    def run(self):
        n = 4 * charm.numPes()
        a = Array(E, n)
        charm.awaitCreation(a)
        g = Group(Probe)
        a.go()
        charm.waitQD()
        ran, killed, present = [], [], []
        for pe in range(charm.numPes()):
            r, k, p = g[pe].state(a.aid, ret=True).get()
            ran += r; killed += k; present += p
        ran.sort(); killed.sort(); present.sort()
        check(len(killed) > 0, 'some pair was local (%d killed)' % len(killed))
        for i in range(0, n, 2):
            check((i in ran) != (i + 1 in ran) or (i not in killed and i + 1 not in killed),
                  'pair (%d, %d): exactly one ran go() when the pair was local' % (i, i + 1))
        check(not set(ran) & set(killed), 'no destroyed element ran go()')
        survivors = [i for i in range(n) if i not in killed]
        check(present == survivors, 'charm.arrays holds exactly the survivors')
        check([a[i].val(ret=True).get() for i in survivors] == survivors, 'the survivors answer')
        print('destroy_sibling: all checks passed')
        charm.exit()


charm.start(Main)

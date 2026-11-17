"""
Constructor numpy arguments: writable where the constructor has its own message.

Run: python tests/thin/ctor_writable.py +pe 1   (and lcrun -n 2 ... +pe 2)

Checks:
- Chare(C, args, onPE), ckInsert, Group and NodeGroup construction: each object
  has its own constructor message, so a numpy argument is writable (a[0] = 1
  works) as for a point-to-point send.
- Array(C, n, args=...): the elements share one message, so the argument is a
  read-only view.
- the sender's array is unchanged.
"""

import numpy as np
from charm4py import charm, Chare, Group, NodeGroup, Array, coro


def check(cond, what):
    if not cond:
        charm.abort('FAIL ' + what)
    print('[PE %d] PASS %s' % (charm.myPe(), what))


class W(Chare):

    def __init__(self, a):
        try:
            a[0] = 1
            self.res = (True, int(a[0]))
        except ValueError:
            self.res = (False, int(a[0]))

    def result(self):
        return self.res


class Main(Chare):

    def __init__(self, args):
        self.thisProxy.run()

    @coro
    def run(self):
        last = charm.numPes() - 1
        arr = np.zeros(8, dtype=np.int64)

        c = Chare(W, args=[arr], onPE=last)
        check(c.result(ret=True).get() == (True, 1), 'Chare(C, args, onPE=%d): argument writable' % last)

        a = Array(W, ndims=1)
        a.ckInsert(0, [arr], onPE=last)
        a.ckInsert(1, [arr], onPE=0)
        a.ckDoneInserting()
        for i in range(2):
            check(a[i].result(ret=True).get() == (True, 1), 'ckInsert element %d: argument writable' % i)

        g = Group(W, args=[arr])
        for pe in range(charm.numPes()):
            check(g[pe].result(ret=True).get() == (True, 1), 'Group branch %d: argument writable' % pe)

        ng = NodeGroup(W, args=[arr])
        for nd in range(charm.numNodes()):
            check(ng[nd].result(ret=True).get() == (True, 1), 'NodeGroup branch %d: argument writable' % nd)

        n = 2 * charm.numPes()
        b = Array(W, n, args=[arr])
        for i in range(n):
            check(b[i].result(ret=True).get() == (False, 0), 'Array(C, %d) element %d: argument read-only' % (n, i))

        check(not arr.any(), "the sender's array is unchanged")
        print('ctor_writable: all checks passed')
        charm.exit()


charm.start(Main)

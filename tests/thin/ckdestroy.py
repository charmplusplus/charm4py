"""
Array elements: ckDestroy on the proxy and on the element.

Run: python -m charmrun.start tests/thin/ckdestroy.py +p1   (and +p2)

Checks:
- proxy[i].ckDestroy() from the main chare on every odd element (on every PE):
  the destroyed elements are gone from charm.arrays on every PE (asked through a
  group), and the even elements still answer.
- proxy[i].ckDestroy() without ret=True is also delivered (polled through the
  group until the element is gone).
- self.ckDestroy() inside an entry method: the element is deleted before the call
  returns (charm._is_local(self) is False and the object is not in charm.arrays);
  the rest of the method still runs and its return value reaches the caller.
- a reduction over the survivors after the destructions: every survivor
  contributes 1 to a sum, which equals the number of survivors.
- proxy.ckDestroy() on the whole array destroys every element.
- a message to a destroyed element is not delivered (Charm++ keeps it for a later
  insertion of the same index): ckInsert of that index receives it.

Checkpoint and restart of the reduced array (not part of the auto_test run):

    python tests/thin/ckdestroy.py +pe 1 ckpt DIR
    python tests/thin/ckdestroy.py +pe 1 ckpt DIR +restart DIR

The first run takes a checkpoint after the destructions; the restarted run checks
that only the survivors exist, that they answer, and repeats the reduction.
"""

from charm4py import charm, Chare, Group, Array, Reducer, coro

ORIGINAL = False   # True only in the process that ran Main.__init__


def check(cond, what):
    if not cond:
        charm.abort('FAIL ' + what)
    print('[PE %d] PASS %s' % (charm.myPe(), what))


class Elem(Chare):

    def __init__(self):
        self.pings = 0

    def val(self):
        return self.thisIndex[0]

    def ping(self):
        self.pings += 1

    def npings(self):
        return self.pings

    def selfdestroy(self):
        self.ckDestroy()
        # the element has been deleted: the rest of this method runs on an object
        # that is no longer in the array
        gone = not charm._is_local(self) and \
            charm.arrays[self.thisProxy.aid].get(self.thisIndex) is not self
        return ('after', gone)

    def count(self, f):
        self.contribute(1, Reducer.sum, f)


class Checker(Chare):

    def present(self, aid):
        """ the indices of array aid that live on this PE """
        return sorted(i[0] for i in charm.arrays.get(aid, {}))


class Main(Chare):

    def __init__(self, args):
        global ORIGINAL
        ORIGINAL = True
        self.ckpt = None
        if 'ckpt' in args:
            self.ckpt = args[args.index('ckpt') + 1]
        self.thisProxy.run()

    def where(self, g, aid):
        return sorted(sum((g[pe].present(aid, ret=True).get() for pe in range(charm.numPes())), []))

    def survivors_ok(self, a, g, survivors, what):
        check(self.where(g, a.aid) == survivors, what + ': charm.arrays holds exactly the survivors ' +
              str(survivors))
        vals = [a[i].val(ret=True).get() for i in survivors]
        check(vals == survivors, what + ': the survivors answer')
        f = charm.Future()
        a.count(f)
        check(f.get() == len(survivors), what + ': a sum reduction over the survivors counts ' +
              str(len(survivors)))

    @coro
    def run(self):
        n = 6 * charm.numPes()
        a = Array(Elem, n)
        charm.awaitCreation(a)
        g = Group(Checker)
        self.a, self.g = a, g
        check(self.where(g, a.aid) == list(range(n)), 'all %d elements exist' % n)

        # destroy every odd element but the last through the proxy and wait
        odd = list(range(1, n - 1, 2))
        for f in [a[i].ckDestroy(ret=True) for i in odd]:
            f.get()
        alive = [i for i in range(n) if i not in odd]
        check(self.where(g, a.aid) == alive, 'proxy ckDestroy removed the odd elements from charm.arrays')

        # fire and forget: poll until the element is gone
        last = n - 1
        a[last].ckDestroy()
        for _ in range(200):
            if last not in self.where(g, a.aid):
                break
            charm.sleep(0.01)
        alive.remove(last)
        check(last not in self.where(g, a.aid), 'proxy ckDestroy without ret removed element %d' % last)

        # destroy from inside an entry method (an even element on the last PE)
        selfi = n - 2
        r = a[selfi].selfdestroy(ret=True).get()
        check(r == ('after', True), 'self.ckDestroy(): the object left before the call returned, '
              'the rest of the method ran (%r)' % (r,))
        alive.remove(selfi)

        self.survivors_ok(a, g, alive, 'after destruction')
        self.alive = alive

        # a message to a destroyed element is kept, not delivered and not an error;
        # inserting the index again delivers it
        a[1].ping()
        charm.sleep(0.2)
        check(1 not in self.where(g, a.aid), 'a message to a destroyed element does not recreate it')
        a.ckInsert(1)
        a.ckDoneInserting()
        check(a[1].npings(ret=True).get() == 1, 'ckInsert of the destroyed index receives the kept message')
        a[1].ckDestroy(ret=True).get()

        # a broadcast destroys every element
        b = Array(Elem, 2 * charm.numPes())
        charm.awaitCreation(b)
        try:
            b.ckDestroy(ret=True)
            check(False, 'proxy.ckDestroy(ret=True) on a whole array raises')
        except Exception as e:
            check('cannot use ret' in str(e), 'proxy.ckDestroy(ret=True) on a whole array raises: ' + str(e))
        b.ckDestroy()
        for _ in range(200):
            if self.where(g, b.aid) == []:
                break
            charm.sleep(0.01)
        check(self.where(g, b.aid) == [], 'proxy.ckDestroy() (broadcast) destroyed every element')

        if self.ckpt is None:
            print('ckdestroy: all checks passed')
            charm.exit()
            return
        f = charm.checkpoint(self.ckpt, resume=self.thisProxy.resumed)
        check(f.get() is None, 'checkpoint of the reduced array written')
        print('ckdestroy: checkpoint run complete')
        charm.exit()

    @coro
    def resumed(self):
        if ORIGINAL:
            return
        self.survivors_ok(self.a, self.g, self.alive, 'after restart')
        print('ckdestroy: restart run complete')
        charm.exit()


charm.start(Main)

"""
Send-path fallbacks and failures.

Run: python tests/thin/send_fallbacks.py +pe 1   (and lcrun -n 2 ... +pe 2)

Checks:
- a str that is not valid UTF-8 (a lone surrogate, as os.fsdecode gives for an
  undecodable file name) is sent by pickling and arrives equal, alone and inside
  a list; plain str, list, dict, tuple and a user object still arrive equal.
- a call whose arguments cannot be pickled (a lambda) raises at the sender, and
  with ret=True or awaitable=True leaves no entry in charm.threadMgr.futures.
"""

from charm4py import charm, Chare, Array, coro


def check(cond, what):
    if not cond:
        charm.abort('FAIL ' + what)
    print('[PE %d] PASS %s' % (charm.myPe(), what))


class Point(object):
    def __init__(self, x):
        self.x = x

    def __eq__(self, o):
        return type(o) is Point and o.x == self.x


class Echo(Chare):

    def echo(self, v):
        return v


class Main(Chare):

    def __init__(self, args):
        self.thisProxy.run()

    @coro
    def run(self):
        last = charm.numPes() - 1
        e = Array(Echo, charm.numPes())
        charm.awaitCreation(e)
        name = 'file\udcff.txt'
        for v in (name, [name, 1], 'plain', [1, 'a'], {'k': (2, 3)}, (4, 5), Point(6)):
            r = e[last].echo(v, ret=True).get()
            check(r == v, 'send to %d of %r arrives equal' % (last, v))

        n0 = len(charm.threadMgr.futures)
        for kw in ('ret', 'awaitable'):
            try:
                e[last].echo(lambda: 0, **{kw: True})
            except Exception as ex:
                check(True, '%s=True with a lambda raises at the sender (%s)' % (kw, type(ex).__name__))
            else:
                charm.abort('FAIL %s=True with a lambda did not raise' % kw)
            check(len(charm.threadMgr.futures) == n0,
                  '%s=True: the failed send left no future (%d entries, %d before)'
                  % (kw, len(charm.threadMgr.futures), n0))
        check(e[last].echo(7, ret=True).get() == 7, 'sends still work after the failures')
        print('send_fallbacks: all checks passed')
        charm.exit()


charm.start(Main)

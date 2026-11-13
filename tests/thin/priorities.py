"""Per-send message priorities, proxy.method(..., prio=p) (SPEC-7-api.md section 3).

Run: python -m charmrun.start tests/thin/priorities.py +p1   (and +p2)

An integer priority, lower runs first (CK_QUEUEING_IFIFO); a message without prio
is queued at priority 0.

Phases:
- local: a group branch, inside a coroutine that has not yet yielded, sends to itself
  100 messages without prio and then one with prio=-1; once the coroutine yields,
  the prio=-1 message runs first and the 100 others follow in send order. Then
  messages sent with priorities 5, 1, None, 3, 10 and one plain send run as None,
  plain (both priority 0, in send order), 1, 3, 5, 10. The same for an array element.
- remote (two PEs or more): PE 0's branch is blocked in a coroutine (a synchronous
  sleep before its first suspension) while the last PE sends it 100 messages without
  prio and then one with prio=-1; all are delivered, and the position at which the
  prio=-1 message ran is printed. It is not required to be first: messages from
  another process enter the priority queue only as the communication backend hands
  them over (SPEC-7-api.md section 6).
- bcast: a broadcast with prio=-1 sent after 20 plain broadcasts reaches every
  branch; on PE 0 (the sender's own PE) it runs first.
- api: prio=None sends without a priority; a non-int prio raises TypeError; prio on
  a section proxy raises Charm4PyError (sections do not carry priorities).
"""
import time
from charm4py import charm, Chare, Group, Array, coro
from charm4py.charm import Charm4PyError

N = 100


def check(cond, what):
    if not cond:
        charm.abort('FAIL ' + what)
    print('[PE %d] PASS %s' % (charm.myPe(), what))


class P(Chare):

    def __init__(self):
        self.order = []

    def recv(self, tag):
        self.order.append(tag)

    def reset(self):
        self.order = []

    def get_order(self):
        return list(self.order)

    def me(self):
        return self.thisProxy[self.thisIndex]

    @coro
    def burst_local(self, n):
        """ 100 plain sends and one prio=-1 send to this object, then yield """
        self.order = []
        me = self.me()
        for i in range(n):
            me.recv(i)
        me.recv('hi', prio=-1)
        self.wait_for(n + 1)
        first = list(self.order)
        self.order = []
        for p in (5, 1, None, 3, 10):
            me.recv(p, prio=p)
        me.recv('plain')
        self.wait_for(6)
        return first, list(self.order)

    def wait_for(self, k):
        while len(self.order) < k:
            charm.sleep(0.001)

    @coro
    def block(self, secs):
        """ busy without yielding (the scheduler does not run), then yield """
        self.order = []
        time.sleep(secs)
        self.wait_for(N + 1)
        return list(self.order)

    def burst_to(self, pe, n):
        dest = self.thisProxy[pe]
        for i in range(n):
            dest.recv(i)
        dest.recv('hi', prio=-1)


class Main(Chare):

    def __init__(self, args):
        self.run()

    @coro
    def run(self):
        g = Group(P)
        a = Array(P, charm.numPes())
        charm.awaitCreation(g, a)
        for proxy, kname in ((g, 'group'), (a, 'array')):
            for pe in sorted({0, charm.numPes() - 1}):
                first, mixed = proxy[pe].burst_local(N, ret=True).get()
                check(first == ['hi'] + list(range(N)),
                      'local %s on PE %d: prio=-1 runs before %d plain messages (first five %r)'
                      % (kname, pe, N, first[:5]))
                check(mixed == [None, 'plain', 1, 3, 5, 10],
                      'local %s on PE %d: priorities 5 1 none 3 10 and a plain send run as %r' % (kname, pe, mixed))
        if charm.numPes() > 1:
            last = charm.numPes() - 1
            f = g[0].block(1.0, ret=True)
            charm.sleep(0.2)   # g[0] is inside its synchronous sleep now
            g[last].burst_to(0, N)
            order = f.get()
            check(sorted(order, key=str) == sorted(list(range(N)) + ['hi'], key=str),
                  'remote: PE %d sent %d plain then one prio=-1 message to blocked PE 0: all delivered' % (last, N))
            # Not checked: on reconverse the prio=-1 message does not reliably run
            # first. Messages still held by the communication backend when PE 0
            # yields reach the priority queue in batches, one batch per progress
            # call, and the queue runs between batches (measured on macOS with LCI:
            # positions 48 to 99 of 101 in six runs with the default scheduler, 24 with
            # +old-scheduler). See SPEC-7-api.md section 6.
            print('[PE 0] INFO remote: the prio=-1 message ran at position %d of %d' % (order.index('hi'), N + 1))
        # bcast
        g.reset(awaitable=True).get()
        for i in range(20):
            g.recv(i)
        g.recv('hi', prio=-1)
        charm.sleep(0.2)
        orders = g.get_order(ret=True).get()
        check(all(sorted(o, key=str) == sorted(list(range(20)) + ['hi'], key=str) for o in orders),
              'bcast: a prio=-1 broadcast reaches every branch')
        check(orders[0][0] == 'hi', 'bcast: on PE 0 the prio=-1 broadcast runs first (%r)' % (orders[0][:3],))
        # api
        g.reset(awaitable=True).get()
        g[0].recv('none', prio=None)
        try:
            g[0].recv('bad', prio='x')
        except TypeError as e:
            check(True, 'api: a non-int prio raises TypeError (%s)' % e)
        else:
            charm.abort('FAIL api: prio="x" accepted')
        try:
            g[0].recv('bad', prio=2**31)
        except OverflowError as e:
            check(True, 'api: a prio outside 32 bits raises OverflowError')
        else:
            charm.abort('FAIL api: prio=2**31 accepted')
        sec = charm.split(a, 1, elems=[[(0,)]])[0]
        try:
            sec.recv('sec', prio=1)
        except Charm4PyError as e:
            check(True, 'api: prio on a section raises Charm4PyError (%s)' % e)
        else:
            charm.abort('FAIL api: prio on a section accepted')
        sec.recv('sec')
        charm.sleep(0.1)
        check(g[0].get_order(ret=True).get() == ['none'], 'api: prio=None sends without a priority')
        print('priorities: all phases passed')
        charm.exit()


charm.start(Main)

"""NodeGroup: one branch per logical node (process), SPEC-7-api.md section 1.

Run: python -m charmrun.start tests/thin/nodegroup.py +p1   (and +p2, +p4)
Restart check, with the same process count, after a run:
     python -m charmrun.start tests/thin/nodegroup.py +pN [dir] +restart <dir>
The default checkpoint directory is charm4py_nodegroup.ckpt in the temporary
directory (tempfile.gettempdir()).

Phases (first run):
- create: NodeGroup(N, args=[...]) from the main coroutine; every branch has
  thisIndex == charm.myNode(), a nodegroup proxy as thisProxy, the constructor
  arguments; charm.numNodes() branches exist; ckLocalBranch on the creating node.
- p2p: ng[node].method(..., ret=True) on every node, from the main chare and from a
  group branch on every PE that received the proxy as an argument.
- bcast: ng.method(...) to every branch, then ng.method(ret=True) gathers one value
  per node in node order.
- reduce: contribute to a future, to a main-chare entry method, and allreduce in a
  coroutine entry of the branches.
- exclusive: in NX (registered for NodeGroup only, as a class with an exclusive entry
  is), every PE sends M increments to an exclusive entry on every node, which does an
  unprotected read-modify-write with a suspension point between the read and the
  write in a coroutine variant. The plain variant counts exactly. The coroutine
  variant shows what exclusive means for a coroutine entry: the lock covers each
  synchronous part (from delivery to the first suspension, and each resume runs on
  its own), so two invocations do not interleave inside a synchronous part, and a
  second invocation does start while the first is suspended.
- checkpoint: branch state is set, charm.checkpoint(dir, resume=Main.resumed) is
  taken; the restarted run checks the restored state of every branch and sends to
  the nodegroup again.
"""
import os
import sys
import tempfile
from charm4py import charm, Chare, Group, NodeGroup, Future, Reducer, coro, entry
from charm4py.chare import GROUP

ORIGINAL = False   # set by Main.__init__: True only in the checkpointing run
M = 50             # exclusive increments per PE per node


def check(cond, what):
    if not cond:
        charm.abort('FAIL ' + what)
    print('[PE %d] PASS %s' % (charm.myPe(), what))


def ckpt_dir(args):
    for a in args[1:]:
        if not a.startswith('+') and not a.endswith('.py'):
            return a
    return os.path.join(tempfile.gettempdir(), 'charm4py_nodegroup.ckpt')


class N(Chare):

    def __init__(self, a, b):
        self.args = (a, b)
        self.node_at_ctor = charm.myNode()
        self.count = 0
        self.state = None

    def info(self):
        return (self.thisIndex, charm.myNode(), charm.myPe(), self.args,
                self.thisProxy.gid, type(self.thisProxy).__name__)

    def bump(self, k):
        self.count += k

    def get_count(self):
        return (self.thisIndex, self.count)

    def sum_to(self, target):
        self.contribute(self.thisIndex + 1, Reducer.sum, target)

    @coro
    def allred(self):
        return self.allreduce(self.thisIndex, Reducer.max).get()

    def set_state(self, v):
        self.state = (v, self.thisIndex)

    def get_state(self):
        return self.state


class NX(Chare):
    """ an exclusive entry makes the class a nodegroup-only class """

    def __init__(self):
        self.n = 0
        self.cn = 0
        self.log = []

    @entry(exclusive=True)
    def incr(self):
        v = self.n
        self.n = v + 1

    @entry(exclusive=True)
    @coro
    def incr_coro(self, tag):
        self.log.append(('start', tag))
        v = self.cn
        charm.sleep(0.01)          # suspension point: the lock is not held here
        self.log.append(('resume', tag))
        self.cn = v + 1

    def counts(self):
        return (self.thisIndex, self.n, self.cn, list(self.log))


class G(Chare):
    """ a group whose branches use a nodegroup proxy they received as an argument """

    def __init__(self):
        pass

    @coro
    def ask(self, ng):
        res = [ng[n].info(ret=True).get()[0] for n in range(charm.numNodes())]
        return res == list(range(charm.numNodes()))

    def flood(self, nx, done):
        for n in range(charm.numNodes()):
            for _ in range(M):
                nx[n].incr()
        self.contribute(None, None, done)

    def flood_coro(self, nx, k):
        for n in range(charm.numNodes()):
            for j in range(k):
                nx[n].incr_coro((charm.myPe(), j))


class Main(Chare):

    def __init__(self, args):
        global ORIGINAL
        ORIGINAL = True
        self.dir = ckpt_dir(args)
        self.got = []
        self.n_done = 0
        self.run()

    def red_target(self, v):
        self.got.append(v)

    @coro
    def run(self):
        nn = charm.numNodes()
        check(nn >= 1 and 0 <= charm.myNode() < nn, 'numNodes %d myNode %d' % (nn, charm.myNode()))
        ng = NodeGroup(N, args=[3, 'x'])
        charm.awaitCreation(ng)
        self.ng = ng
        # create
        infos = [ng[n].info(ret=True).get() for n in range(nn)]
        for n, (idx, node, pe, args, gid, pname) in enumerate(infos):
            check(idx == n and node == n and args == (3, 'x') and gid == ng.gid and pname == 'NNodeGroupProxy',
                  'create: branch %d index %d on node %d PE %d args %r proxy %s' % (n, idx, node, pe, args, pname))
        check(charm.nodegroups[ng.gid] is ng.ckLocalBranch() and ng.ckLocalBranch().thisIndex == charm.myNode(),
              'create: ckLocalBranch and charm.nodegroups')
        check(ng[0].elemIdx == 0 and ng.elemIdx == -1, 'create: elemIdx of an element and of the whole nodegroup')
        try:
            ng[0:1]
        except Exception as e:
            check(True, 'create: a slice of a nodegroup proxy raises (%s)' % type(e).__name__)
        else:
            charm.abort('FAIL create: slice of a nodegroup proxy accepted')
        # p2p from a group on every PE
        g = Group(G)
        charm.awaitCreation(g)
        oks = g.ask(ng, ret=True).get()
        check(oks == [True] * charm.numPes(), 'p2p: every PE reaches every node through a proxy argument')
        # bcast
        ng.bump(2)
        ng.bump(5)
        counts = ng.get_count(ret=True).get()
        check(counts == [(n, 7) for n in range(nn)], 'bcast: two broadcasts reached every branch: %r' % (counts,))
        # reduce
        f = Future()
        ng.sum_to(f)
        check(f.get() == nn * (nn + 1) // 2, 'reduce: sum to a future')
        ng.sum_to(self.thisProxy.red_target)
        while not self.got:
            charm.sleep(0.01)
        check(self.got == [nn * (nn + 1) // 2], 'reduce: sum to a main-chare entry method: %r' % (self.got,))
        r = ng.allred(ret=True).get()
        check(r == [nn - 1] * nn, 'reduce: allreduce in a coroutine entry: %r' % (r,))
        # exclusive
        nx = NodeGroup(NX)
        charm.awaitCreation(nx)
        check(N in charm.proxyClasses[GROUP] and NX not in charm.proxyClasses[GROUP],
              'exclusive: N is a group class too, NX is not')
        done = Future()
        g.flood(nx, done)
        done.get()
        charm.waitQD()
        c = nx.counts(ret=True).get()
        check([x[1] for x in c] == [M * charm.numPes()] * nn,
              'exclusive: plain exclusive entry counted %r of %d per node' % ([x[1] for x in c], M * charm.numPes()))
        k = 3
        g.flood_coro(nx, k)
        # charm.sleep timers are not counted by quiescence detection: poll until
        # every invocation has resumed
        for _ in range(500):
            c = nx.counts(ret=True).get()
            if all(sum(1 for e in x[3] if e[0] == 'resume') == k * charm.numPes() for x in c):
                break
            charm.sleep(0.01)
        for node, n, cn, log in c:
            starts = [i for i, e in enumerate(log) if e[0] == 'start']
            first_resume = min(i for i, e in enumerate(log) if e[0] == 'resume')
            check(len(starts) == k * charm.numPes() and cn <= len(starts),
                  'exclusive coroutine node %d: %d invocations ran, counter %d' % (node, len(starts), cn))
            check(len(starts) == 1 or starts[1] < first_resume,
                  'exclusive coroutine node %d: a second invocation starts while the first is suspended' % node)
        # checkpoint
        ng.set_state(42)
        ng.get_state(ret=True).get()
        f = charm.checkpoint(self.dir, resume=self.thisProxy.resumed)
        check(f.get() is None, 'checkpoint: future deposited (dir %s)' % self.dir)
        self.one_done()

    def one_done(self):
        self.n_done += 1
        if self.n_done == 2:
            print('nodegroup: all phases passed')
            charm.exit()

    @coro
    def resumed(self):
        if ORIGINAL:
            check(True, 'checkpoint: resume entry called in the checkpointing run')
            self.one_done()
            return
        nn = charm.numNodes()
        st = self.ng.get_state(ret=True).get()
        check(st == [(42, n) for n in range(nn)], 'restart: nodegroup state restored: %r' % (st,))
        counts = self.ng.get_count(ret=True).get()
        check(counts == [(n, 7) for n in range(nn)], 'restart: counters restored: %r' % (counts,))
        self.ng.bump(1)
        infos = [self.ng[n].info(ret=True).get() for n in range(nn)]
        check([i[0] for i in infos] == list(range(nn)) and all(i[3] == (3, 'x') for i in infos),
              'restart: point-to-point sends after the restart')
        check(charm.nodegroups[self.ng.gid].thisIndex == charm.myNode(), 'restart: charm.nodegroups has the branch')
        print('nodegroup: restart run passed')
        charm.exit()


charm.start(Main)

"""
Checkpoint and restart with charm.checkpoint(dirname, resume).

Run it twice with the same PE count and checkpoint directory:

    python tests/thin/checkpoint_restart.py +pe 1 [dir]
    python tests/thin/checkpoint_restart.py +pe 1 [dir] +restart <dir>

(with two processes: lcrun -n 2 python ... +pe 2, the second run adding
+restart <dir>). The default dir is ./checkpoint_restart.ckpt.

First run: the main chare's coroutine sets state in itself, a group, an array
and a readonly, calls charm.checkpoint(dir, resume=self.thisProxy.resumed),
waits on the returned future, and exits once both the future and the resume
entry (which the runtime also calls in the checkpointing run) have arrived.

Second run (+restart): the main chare's constructor and coroutine are not run;
the runtime restores the objects and calls Main.resumed, which checks the
restored state on every PE, then takes a second checkpoint with a group entry
as the resume target (one delivery per PE, the future on PE 0) and exits.
Each check prints a PASS line; any failure aborts.
"""

import os
import sys
from charm4py import charm, Chare, Group, Array, coro, readonlies as ro

ORIGINAL = False   # set by Main.__init__: True only in the checkpointing process


def ckpt_dir(args):
    for a in args[1:]:
        if not a.startswith('+') and not a.endswith('.py'):
            return a
    return 'checkpoint_restart.ckpt'


def check(cond, what):
    if not cond:
        charm.abort('FAIL ' + what)
    print('[PE %d] PASS %s' % (charm.myPe(), what))


class G(Chare):

    def __init__(self):
        self.val = 100 + self.thisIndex

    def bump(self):
        self.val += 1000

    def get(self):
        return (charm.myPe(), self.val)

    def resumed2(self):
        # resume target of the second checkpoint (taken in the restarted run)
        self.seen2 = True


class A(Chare):

    def __init__(self):
        self.val = self.thisIndex[0] * 10

    def bump(self):
        self.val += 1

    def get(self):
        return self.val


class Main(Chare):

    def __init__(self, args):
        global ORIGINAL
        ORIGINAL = True
        self.dir = ckpt_dir(args)
        ro.magic = 1234
        self.counter = 0
        self.n_done = 0
        self.g = Group(G)
        self.a = Array(A, 2 * charm.numPes())
        self.thisProxy.run()

    @coro
    def run(self):
        self.counter = 42
        self.g.bump(ret=True).get()
        self.a.bump(ret=True).get()
        f = charm.checkpoint(self.dir, resume=self.thisProxy.resumed)
        check(f is not None, 'charm.checkpoint from a coroutine returns a future')
        check(f.get() is None, 'checkpoint future deposited in the checkpointing run')
        self.one_done()

    def one_done(self):
        self.n_done += 1
        if self.n_done == 2:
            print('checkpoint run complete')
            charm.exit()

    @coro
    def resumed(self):
        if ORIGINAL:
            check(True, 'resume entry called in the checkpointing run')
            self.one_done()
            return
        check(self.counter == 42, 'main chare state restored (counter 42)')
        check(ro.magic == 1234, 'readonly restored (magic 1234)')
        vals = self.g.get(ret=True).get()
        check(sorted(vals) == [(pe, 1100 + pe) for pe in range(charm.numPes())],
              'group state restored on every PE: %r' % (vals,))
        n = 2 * charm.numPes()
        avals = [self.a[i].get(ret=True).get() for i in range(n)]
        check(avals == [i * 10 + 1 for i in range(n)], 'array state restored: %r' % (avals,))
        # a second checkpoint with a group entry as the resume target
        f = charm.checkpoint(self.dir + '.2', resume=self.g.resumed2)
        check(f.get() is None, 'second checkpoint (group resume target) future deposited')
        print('restart run complete')
        charm.exit()


charm.start(Main)

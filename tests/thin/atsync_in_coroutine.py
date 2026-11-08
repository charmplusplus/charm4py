"""Migration and load-balancing checks of step 4c (SPEC-4c.md, DESIGN.md 4.5 items 1-2,
5.4 item 14, review F9 and F10).

Run: lcrun -n 2 python tests/thin/atsync_in_coroutine.py +pe 2 +balancer RotateLB
     (any process count >= 2; RotateLB moves every migratable element every period)
     python tests/thin/atsync_in_coroutine.py norecurse +pe 1   (no balancer)

Phase 1. A coroutine entry calls AtSync() and returns. Every element must change PE
   at least once over PERIODS load-balancing periods: an element whose coroutine
   called AtSync is not left non-migratable (F9). Each element also checks after
   every period that its state was restored: a numpy attribute, thisIndex,
   thisProxy, _ckdyn_target, and an attribute that __getstate__ drops and
   __setstate__ recreates. A message buffered behind a @when condition before the
   period is delivered after it, on the new PE (the wait queue is part of the state).
Phase 2. Even elements hold a coroutine suspended in self.wait(...); for them
   migrate() raises Charm4PyError, and over two periods the balancer leaves them on
   their PE while the odd elements move. After the coroutines are released, one more
   period moves every element.
Mode 'norecurse' (run without +balancer). resumeFromSync calls AtSync again
   directly, NITER times. resumeFromSync is delivered by a message, so the Python
   stack does not grow (F10); a synchronous call would raise RecursionError.
"""
from charm4py import charm, Chare, Array, Reducer, Future, coro, when
from charm4py.charm import Charm4PyError
import numpy as np
import sys
import threading

PERIODS = 4
NITER = 3000


class Elem(Chare):

    def __init__(self):
        self.data = np.arange(1000, dtype='float64') * (self.thisIndex[0] + 1)
        self.pes = [charm.myPe()]
        self.lock = threading.Lock()   # not picklable: dropped by __getstate__
        self.period = 0
        self.got = []
        self.phase = 1

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['lock']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.lock = threading.Lock()

    def check_state(self):
        i = self.thisIndex[0]
        assert isinstance(self.thisIndex, tuple) and len(self.thisIndex) == 1
        np.testing.assert_array_equal(self.data, np.arange(1000, dtype='float64') * (i + 1))
        assert tuple(self._ckdyn_target.index) == self.thisIndex, (self._ckdyn_target.index, self.thisIndex)
        assert self.thisProxy == self.arr, 'thisProxy changed'
        assert self.lock.acquire(blocking=False)
        self.lock.release()

    @coro
    def step(self, arr, done):
        # a coroutine that calls AtSync and returns (particle.py's pattern)
        self.arr = arr
        self.done = done
        # a message to the next period, buffered behind the when condition until
        # resumeFromSync advances self.period
        self.thisProxy[self.thisIndex].tagged(self.period + 1, self.data[:4].copy())
        self.AtSync()

    @when('self.period == p')
    def tagged(self, p, x):
        self.got.append((p, charm.myPe()))
        np.testing.assert_array_equal(x, self.data[:4])
        if p == PERIODS:
            self.contribute((self.thisIndex[0], self.pes, self.got), Reducer.gather, self.done)
        else:
            self.thisProxy[self.thisIndex].step(self.arr, self.done)

    def resumeFromSync(self):
        self.check_state()
        if self.phase == 2:
            self.contribute((self.thisIndex[0], self.before, charm.myPe(), self._nsuspended),
                            Reducer.gather, self.f)
            return
        self.pes.append(charm.myPe())
        self.period += 1   # releases the buffered 'tagged' message

    # ---- phase 2 ----

    @coro
    def block(self, ready):
        self.released = False
        self.contribute(None, None, ready)
        self.wait('self.released')

    def try_migrate(self):
        try:
            self.migrate((charm.myPe() + 1) % charm.numPes())
        except Charm4PyError as e:
            assert 'suspended coroutine' in str(e), str(e)
            return True
        return False

    def sync(self, f):
        self.f = f
        self.before = charm.myPe()
        self.AtSync()

    def release(self):
        self.released = True

    def start_phase2(self):
        self.phase = 2

    def contribute_none(self, f):
        self.contribute(None, None, f)


class Base(Chare):

    def __init__(self, x):
        self.x = x


class Sub(Base):
    # a chare class that inherits its constructor from another chare class: the
    # runtime runs Chare.__init__ first, then Base.__init__ once with the arguments

    def get(self):
        return self.x, self._nsuspended


class Rec(Chare):

    def go(self, n, f):
        self.n = n
        self.i = 0
        self.f = f
        self.depth0 = len(_stack())
        self.AtSync()

    def resumeFromSync(self):
        self.i += 1
        d = len(_stack())
        assert d <= self.depth0 + 5, 'resumeFromSync nested: stack depth %d, first %d' % (d, self.depth0)
        if self.i == self.n:
            self.contribute(None, None, self.f)
        else:
            self.AtSync()


def _stack():
    import inspect
    return inspect.stack(0)


def norecurse():
    a = Array(Rec, charm.numPes() * 2, useAtSync=True)
    f = Future()
    a.go(NITER, f)
    f.get()
    print('atsync_in_coroutine norecurse: PASS (%d AtSync calls per element)' % NITER)


def main(args):
    if 'norecurse' in args:
        norecurse()
        exit()
    npes = charm.numPes()
    assert npes >= 2, 'run with at least 2 PEs (one per process)'
    n = npes * 4

    # phase 1
    a = Array(Elem, n, useAtSync=True)
    done = Future()
    a.step(a, done)
    res = sorted(done.get())
    for i, pes, got in res:
        assert len(pes) == PERIODS + 1, (i, pes)
        assert len(set(pes)) > 1, 'element %d never changed PE: %s' % (i, pes)
        assert [p for p, _ in got] == list(range(1, PERIODS + 1)), (i, got)
        # each buffered message ran on the PE the element had after the period
        assert [pe for _, pe in got] == pes[1:], (i, got, pes)
    print('phase 1: %d elements, PEs per element over %d periods, e.g. %s' % (n, PERIODS, res[0][1]))

    # phase 2, on the same array: every element with useAtSync on a PE takes part in
    # that PE's load-balancing barrier, so a second AtSync array would wait for this one
    b = a
    b.start_phase2(awaitable=True).get()
    ready = Future()
    for i in range(n):
        if i % 2 == 0:
            b[i].block(ready)
        else:  # odd elements contribute directly, so that 'ready' gets n contributions
            b[i].contribute_none(ready)
    ready.get()
    # only the suspended (even) elements: an element moved by migrate() and then
    # taking part in the next AtSync step hangs that step (runtime defect, also in
    # plain C++ on recharm: logs/gate-4c/cxx-repro, SPEC-4c.md 6)
    raised = [b[i].try_migrate(ret=True).get() for i in range(0, n, 2)]
    assert all(raised), raised
    for period in range(2):
        f = Future()
        b.sync(f)
        for i, before, after, nsusp in sorted(f.get()):
            if i % 2 == 0:
                assert nsusp == 1 and before == after, ('suspended element moved', i, before, after)
            else:
                assert nsusp == 0 and before != after, ('migratable element did not move', i, before, after)
    # the coroutines resume when release() runs (auto_flush_wait_queues), before
    # the next message to the element
    b.release(awaitable=True).get()
    f = Future()
    b.sync(f)
    for i, before, after, nsusp in sorted(f.get()):
        assert nsusp == 0 and before != after, ('element did not move after release', i, before, after)
    print('phase 2: suspended elements stayed and refused migrate(); all moved after release')
    s = Array(Sub, 2, args=[7])
    assert s[1].get(ret=True).get() == (7, 0)
    print('atsync_in_coroutine: PASS')
    exit()

charm.start(main)

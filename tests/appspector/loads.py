"""charm4py.appspector: measured times and the encoded replies.

Element i of an array spins (i + 1) * 2 ms per step, called by broadcast
(per-element attribution inside a broadcast) and from a coroutine that
resumes after a future (time after a resume charged to its entry method).
Then a gather is taken the way a CCS request takes it and checked.
"""
import os
import struct
import sys
import time

from charm4py import charm, Chare, Array, Group, Reducer, coro, appspector, loadfmt
from charm4py.charm import Charm4PyError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import state     # the parts that need no running job

UNIT = 0.002
STEPS = 20


def spin(seconds):
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        pass


class Elem(Chare):

    def work(self):
        spin(UNIT * (self.thisIndex[0] + 1))

    @coro
    def resumed(self, steps, done):
        for _ in range(steps):
            self.allreduce(1, Reducer.sum).get()
            spin(UNIT * (self.thisIndex[0] + 1))
        self.reduce(done, None, Reducer.nop)


class Probe(Chare):

    def serve_here(self):
        try:
            appspector.serve()
        except Charm4PyError:
            return True
        return False


def gather(proxy):
    # the way a request gathers, without a CCS client
    r = charm.groups[proxy.gid].requests
    r.last = None
    r.gen += 1
    proxy.report(r.gen)
    while r.last is None:
        charm.sleep(0.01)
    return r.last


def main(args):
    state.run()
    proxy = appspector.serve()
    assert appspector.serve() is proxy       # a second call does nothing
    assert Group(Probe)[1].serve_here(ret=True).get()   # refused on PE 1
    n = charm.numPes() * 2
    elems = Array(Elem, n)
    charm.awaitCreation(elems)
    before = gather(proxy)
    for _ in range(STEPS):
        elems.work(awaitable=True).get()
    done = charm.Future()
    elems.resumed(STEPS, done)
    done.get()
    after = gather(proxy)
    # the gather timer through the wrapped triggerCallable (a stale gather
    # number: does nothing) is not charged as load
    r = charm.groups[proxy.gid].requests
    timer_before = r.last[0][5].get(loadfmt.TIMER_EP, 0.0)
    charm.scheduleCallableAfter(r.timed_out, 0.01, [0])
    charm.sleep(0.05)
    assert not r.gathering and not r.waiting
    assert charm.groups[proxy.gid].tracker.eps.get(loadfmt.TIMER_EP, 0.0) <= timer_before + 0.04
    names = {ep: '%s.%s' % (em.C.__name__, em.name) for ep, em in charm.entryMethods.items()}
    # a class has one entry method per collection type it is registered for
    ep_work = {ep for ep, nm in names.items() if nm == 'Elem.work'}
    ep_res = {ep for ep, nm in names.items() if nm == 'Elem.resumed'}
    aid = elems.aid
    times = {}
    for pe, now, elapsed, idle, objs, eps in after:
        assert 0 <= idle <= elapsed
        for om, oid, secs, mig in objs:
            if om == aid:
                assert mig
                times[oid] = secs
    assert sorted(times) == list(range(n)), times
    for i in range(n):
        expect = 2 * STEPS * UNIT * (i + 1)          # work + resumed
        assert expect <= times[i] < expect * 1.5 + 0.02, (i, times[i], expect)
    t_work = sum(t for p in after for ep, t in p[5].items() if ep in ep_work)
    t_res = sum(t for p in after for ep, t in p[5].items() if ep in ep_res)
    total = STEPS * UNIT * n * (n + 1) / 2
    assert total <= t_work < total * 1.5 + 0.05, (t_work, total)
    assert total <= t_res < total * 1.5 + 0.05, (t_res, total)
    reply = loadfmt.encode_objloads(p[:5] for p in after)
    assert struct.unpack_from('<i', reply)[0] == charm.numPes()
    print('appspector loads: OK (%d elements, work %.3f s, resumed %.3f s, expected %.3f s each)'
          % (n, t_work, t_res, total))
    charm.exit()


charm.start(main)

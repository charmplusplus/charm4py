"""charm4py.appspector: measured times and the encoded replies.

Element i of an array spins (i + 1) * 2 ms per call, called by broadcast
and through a section over the array (per-element attribution inside
both), and from a coroutine that resumes after a future (time after a
resume charged to its entry method). Then a gather is taken the way a CCS
request takes it and checked.

Each element also times its own calls from inside, and the checks are
against that rather than against the spin lengths: a PE descheduled in
the middle of a spin (a busy CI machine) lengthens a call but also the
element's own measurement of it. The time charged to an element must be
at least its own measurement (exact: those intervals lie inside the calls
appspector times, so time charged to the wrong chare fails it), at most
twice that (a call's delivery outside the element's own measurement is
short, but a busy machine can deschedule the PE there too), and no PE's
chares may be charged more than its busy time (nothing counted twice).
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

    def __init__(self):
        self.own = {'work': 0.0, 'resumed': 0.0}   # seconds, measured inside the calls

    def work(self):
        t = time.perf_counter()
        spin(UNIT * (self.thisIndex[0] + 1))
        self.own['work'] += time.perf_counter() - t

    @coro
    def resumed(self, steps, done):
        t = time.perf_counter()
        for _ in range(steps):
            f = self.allreduce(1, Reducer.sum)
            self.own['resumed'] += time.perf_counter() - t
            f.get()
            t = time.perf_counter()
            spin(UNIT * (self.thisIndex[0] + 1))
        self.reduce(done, None, Reducer.nop)
        self.own['resumed'] += time.perf_counter() - t

    def own_times(self):
        return self.thisIndex[0], self.own['work'], self.own['resumed']


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
    section = elems[0:n]        # delivered by the section manager group
    before = gather(proxy)
    for _ in range(STEPS):
        elems.work(awaitable=True).get()
        section.work(awaitable=True).get()
    done = charm.Future()
    elems.resumed(STEPS, done)
    done.get()
    after = gather(proxy)
    own = {i: (w, r) for i, w, r in elems.own_times(ret=True).get()}
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
    times = {}                  # element -> seconds charged between the two gathers
    for snaps, sign in ((before, -1), (after, 1)):
        for pe, now, elapsed, idle, objs, eps in snaps:
            assert 0 <= idle <= elapsed
            # exclusive times: no chare is charged time another one is also charged
            assert sum(o[2] for o in objs) <= elapsed - idle + 1e-6, (pe, objs)
            for om, oid, secs, mig in objs:
                if om == aid:
                    assert mig
                    times[oid] = times.get(oid, 0.0) + sign * secs
    assert sorted(times) == list(range(n)), times
    for i in range(n):
        w, res = own[i]
        expect = 3 * STEPS * UNIT * (i + 1)          # work twice, resumed
        assert expect <= w + res <= times[i] + 1e-9 <= 2 * (w + res) + 0.05, \
            (i, times[i], w + res, expect)
    t_work = sum(t for p in after for ep, t in p[5].items() if ep in ep_work)
    t_res = sum(t for p in after for ep, t in p[5].items() if ep in ep_res)
    own_work = sum(w for w, res in own.values())
    own_res = sum(res for w, res in own.values())
    assert own_work <= t_work + 1e-9 <= 2 * own_work + 0.05, (t_work, own_work)
    assert own_res <= t_res + 1e-9 <= 2 * own_res + 0.05, (t_res, own_res)
    reply = loadfmt.encode_objloads(p[:5] for p in after)
    assert struct.unpack_from('<i', reply)[0] == charm.numPes()
    print('appspector loads: OK (%d elements, work %.3f s for %.3f s measured inside, '
          'resumed %.3f s for %.3f s)' % (n, t_work, own_work, t_res, own_res))
    charm.exit()


charm.start(main)

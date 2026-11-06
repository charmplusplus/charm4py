"""charm4py.appspector without a running job: the reply formats, the
per-PE counters (pruning, install once, section deliveries) and PE 0's
request state (gather timeout, late gathers, freshness of a gather reused
for SumDetail).

    python tests/appspector/state.py

(with an installed charm4py; nothing is started). loads.py runs it too.
"""
import struct

from greenlet import getcurrent

from charm4py import loadfmt, appspector
from charm4py.charm import Charm4PyError


def test_format():
    assert loadfmt.obj_id((7,)) == 7
    assert loadfmt.obj_id((1, 2)) == (1 << 32) | 2
    assert loadfmt.obj_id(('a', 1)) == loadfmt.obj_id(('a', 1))
    data = loadfmt.encode_objloads([(1, 10.0, 5.0, 2.0, [(3, 4, 1.5, True)])])
    assert struct.unpack('<iidddiQidB', data) == (1, 1, 10.0, 5.0, 2.0, 1, 4, 3, 1.5, 1)
    assert loadfmt.encode_objloads([]) == struct.pack('<i', 0)
    items = loadfmt.ep_percentages({0: (0.0, {5: 0.0}), 1: (0.0, {})},
                                   {0: (1.0, {5: 0.5}), 1: (1.0, {5: 0.25})}, 2)
    assert items == [(5, 37.5)], items
    assert loadfmt.encode_sumdetail(2, [items]) == struct.pack('<iihhB', 1, 2, 1, 5, 94)
    assert loadfmt.encode_sumdetail(2, []) == struct.pack('<i', 0)
    assert 'ENTRY CHARE 9 "A.b" 0 0' in loadfmt.sts_text({9: 'A.b'})


def test_snapshot_prunes():
    t = appspector.Tracker(getcurrent())
    t.objs = {('a', 1, (0,)): 1.0, ('a', 1, (1,)): 2.0, ('a', 2, (0,)): 3.0,
              ('g', 5): 4.0, ('g', 6): 5.0,
              ('c', (0, 77)): 6.0, ('c', (0, 78)): 7.0, ('c', None): 8.0}
    arrays = {1: {(0,): object()}}
    groups = {5: object()}
    chares = {(0, 77): object()}
    pe, now, elapsed, idle, objs, eps = t.snapshot(3, arrays, groups, chares)
    assert pe == 3 and 0 <= idle <= elapsed
    assert sorted(objs) == sorted([(1, 0, 1.0, True), (5, 3, 4.0, False),
                                   (-1, loadfmt.chare_obj_id((0, 77)), 6.0, False)]), objs
    assert set(t.objs) == {('a', 1, (0,)), ('g', 5), ('c', (0, 77))}, t.objs


class FakeCharm(object):
    def __init__(self):
        self.sched_callables = {}
        self.calls = []

    def recvArrayMsg(self, *a): self.calls.append('a')
    def recvGroupMsg(self, *a): self.calls.append('g')
    def recvChareMsg(self, *a): self.calls.append('c')
    def recvArrayBcast(self, *a): self.calls.append('b')
    def invokeEntryMethod(self, *a, **k): self.calls.append('i')

    def triggerCallable(self, tag):
        cb, args = self.sched_callables.pop(tag)
        cb(*args)


class FakeThreads(object):
    def resumeThread(self, gr, arg): pass
    def pauseThread(self): pass


def test_install_once():
    c, tm = FakeCharm(), FakeThreads()
    t = appspector.Tracker(getcurrent(), skip_gids=[9])
    t.install(c, tm)
    try:
        appspector.Tracker(getcurrent()).install(c, tm)
    except Charm4PyError:
        pass
    else:
        raise AssertionError('second install did not raise')
    c.recvGroupMsg(4, 7, b'', 0)
    c.recvGroupMsg(9, 7, b'', 0)      # our own group: not measured
    assert c.calls == ['g', 'g'] and set(t.objs) == {('g', 4)} and set(t.eps) == {7}

    class Owner(object):
        def fire(self, out):
            out.append(1)
    quiet, loud, out = Owner(), Owner(), []
    t.quiet = quiet
    c.sched_callables[1] = (quiet.fire, [out])
    c.triggerCallable(1)
    assert loadfmt.TIMER_EP not in t.eps
    c.sched_callables[2] = (loud.fire, [out])
    c.triggerCallable(2)
    assert loadfmt.TIMER_EP in t.eps and out == [1, 1]


class Clock(object):
    t = 100.0

    def __call__(self):
        return self.t


def test_section_fanout():
    # A section message reaches the section manager group (gid 3), whose
    # entry method hands it on to each member: each member's call must be
    # charged to that member, the manager keeping only its own time.
    class Proxy(object):
        def __init__(self, **ids):
            self.__dict__.update(ids)

    class Obj(object):
        def __init__(self, proxy, index=None):
            self.thisProxy, self.thisIndex = proxy, index

    clock = Clock()
    manager = Obj(Proxy(gid=3))
    members = [Obj(Proxy(aid=5), (i,)) for i in range(3)]

    class SectionCharm(FakeCharm):
        def recvGroupMsg(self, gid, ep, msg, dcopy_start):
            self.invokeEntryMethod(manager, ep, None, ())     # as charm.recvGroupMsg
        def invokeEntryMethod(self, obj, ep, header, args, ret_fut=False):
            if obj is manager:                                 # SectionManager.sendToSection
                clock.t += 0.001
                for m in members:
                    self.invokeEntryMethod(m, 8, None, ())
            else:                                              # a member's entry method
                clock.t += 0.01 * (obj.thisIndex[0] + 1)

    saved = appspector._clock
    appspector._clock = clock
    try:
        c = SectionCharm()
        t = appspector.Tracker(getcurrent(), fanout_gids=[3])
        t.install(c, FakeThreads())
        c.recvGroupMsg(3, 7, b'', 0)
    finally:
        appspector._clock = saved
    want = {('a', 5, (0,)): 0.01, ('a', 5, (1,)): 0.02, ('a', 5, (2,)): 0.03, ('g', 3): 0.001}
    assert set(t.objs) == set(want), t.objs
    assert all(abs(t.objs[k] - v) < 1e-9 for k, v in want.items()), t.objs
    assert abs(t.eps[8] - 0.06) < 1e-9 and abs(t.eps[7] - 0.001) < 1e-9, t.eps
    assert abs(t.busy - 0.061) < 1e-9, t.busy


class Harness(object):
    """Requests with recorded side effects instead of charm calls."""

    def __init__(self, num_pes=2):
        self.clock = Clock()
        self.gathers = []        # gens asked for
        self.timers = []         # (fn, secs, args)
        self.delayed = {}        # token -> list of replies sent
        self.direct = []         # immediate replies
        self.ntok = 0
        self.r = appspector.Requests(num_pes, self.gathers.append, self.delay,
                                     self.send_delayed, self.direct.append,
                                     self.schedule, clock=self.clock)

    def delay(self):
        self.ntok += 1
        self.delayed[self.ntok] = []
        return self.ntok

    def send_delayed(self, token, msg):
        self.delayed[token].append(msg)

    def schedule(self, fn, secs, args):
        self.timers.append((fn, secs, args))

    def fire_timers(self):
        timers, self.timers = self.timers, []
        for fn, secs, args in timers:
            fn(*args)

    def snaps(self, gen, now, ep_secs):
        return [(gen, (pe, now, now, 0.0, [], {5: ep_secs})) for pe in range(self.r.num_pes)]


def objloads_pes(msg):
    return struct.unpack_from('<i', msg)[0]


def sumdetail_bins(msg):
    return struct.unpack_from('<i', msg)[0]


def test_gather_and_reuse():
    h = Harness()
    r = h.r
    r.request('obj')
    assert h.gathers == [1] and r.gathering and h.timers[0][1] == appspector.GATHER_TIMEOUT
    r.request('obj')                        # joins the outstanding gather
    assert h.gathers == [1]
    r.gathered(h.snaps(1, 1.0, 0.0))
    assert not r.gathering
    assert [objloads_pes(m[0]) for m in h.delayed.values()] == [2, 2]
    r.sumdetail()                           # reuses the ObjLoads gather: first one, nothing yet
    assert h.gathers == [1] and sumdetail_bins(h.direct[-1]) == 0
    r.sumdetail()                           # gather used already: a new one
    assert h.gathers == [1, 2]
    h.clock.t += 1.0
    r.gathered(h.snaps(2, 2.0, 0.5))
    msg = h.delayed[3][0]
    assert struct.unpack('<iihhB', msg) == (1, 2, 1, 5, 125), msg   # 50 %
    h.fire_timers()                         # both gathers came back: no effect
    assert all(len(v) == 1 for v in h.delayed.values())


def test_freshness():
    h = Harness()
    r = h.r
    r.request('obj')
    r.gathered(h.snaps(1, 1.0, 0.0))
    r.sumdetail()                           # sets ep_prev
    r.request('obj')
    h.clock.t += 0.5
    r.gathered(h.snaps(2, 2.0, 0.25))
    h.clock.t += appspector.FRESH_SECONDS   # ObjLoads gather is now stale
    r.sumdetail()
    assert h.gathers == [1, 2, 3], h.gathers
    r.gathered(h.snaps(3, 3.0, 0.5))
    assert struct.unpack('<iihhB', h.delayed[3][0]) == (1, 2, 1, 5, 62)  # 0.5 s of 2 s = 25 %
    r.request('obj')
    h.clock.t += 0.2
    r.gathered(h.snaps(4, 3.5, 0.5))
    r.sumdetail()                           # recent: answered at once
    assert h.gathers == [1, 2, 3, 4] and len(h.direct) == 2


def test_timeout_and_late_gather():
    h = Harness()
    r = h.r
    r.request('obj')
    r.request('ep')
    h.fire_timers()                         # no PE answered, no gather yet
    assert not r.gathering
    assert objloads_pes(h.delayed[1][0]) == 0 and sumdetail_bins(h.delayed[2][0]) == 0
    r.request('obj')                        # a new gather starts
    assert h.gathers == [1, 2] and r.gathering
    r.gathered(h.snaps(1, 1.0, 0.0))        # the late first one: data, no reply
    assert r.gathering and len(h.delayed[3]) == 0 and r.last is not None
    r.gathered(h.snaps(2, 2.0, 0.0))
    assert not r.gathering and objloads_pes(h.delayed[3][0]) == 2
    h.fire_timers()                         # gen 2's timer: already answered
    assert all(len(v) == 1 for v in h.delayed.values()), h.delayed
    # a timeout with an older gather answers ObjLoads from it
    r.request('obj')
    r.request('ep')
    h.fire_timers()
    assert objloads_pes(h.delayed[4][0]) == 2
    # gather 2 was not used for SumDetail yet: answered from it (the first
    # SumDetail interval, so "nothing yet", but it now starts the next one)
    assert sumdetail_bins(h.delayed[5][0]) == 0 and r.ep_prev is not None and not r.ep_fresh
    r.request('ep')
    h.fire_timers()
    assert sumdetail_bins(h.delayed[6][0]) == 0     # nothing new: "nothing yet"
    # mixed generations (broadcasts overtaking): wait for the newest
    r.request('obj')
    gen = h.gathers[-1]
    r.gathered([(gen - 1, (0, 1.0, 1.0, 0.0, [], {})), (gen, (1, 1.0, 1.0, 0.0, [], {}))])
    assert r.gathering and len(h.delayed[7]) == 0
    r.gathered(h.snaps(gen, 5.0, 1.0))
    assert not r.gathering and len(h.delayed[7]) == 1


def run():
    test_format()
    test_snapshot_prunes()
    test_install_once()
    test_section_fanout()
    test_gather_and_reuse()
    test_freshness()
    test_timeout_and_late_gather()


if __name__ == '__main__':
    run()
    print('appspector state: OK')

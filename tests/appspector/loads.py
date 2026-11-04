"""charm4py.appspector: measured times and the encoded replies.

Element i of an array spins (i + 1) * 2 ms per step, called by broadcast
(per-element attribution inside a broadcast) and from a coroutine that
resumes after a future (time after a resume charged to its entry method).
Then a gather is taken the way a CCS request takes it and checked.
"""
import struct
import time

from charm4py import charm, Chare, Array, Reducer, coro, appspector, loadfmt

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


def test_format():
    assert loadfmt.obj_id((7,)) == 7
    assert loadfmt.obj_id((1, 2)) == (1 << 32) | 2
    assert loadfmt.obj_id(('a', 1)) == loadfmt.obj_id(('a', 1))
    data = loadfmt.encode_objloads([(1, 10.0, 5.0, 2.0, [(3, 4, 1.5, True)])])
    assert struct.unpack('<iidddiQidB', data) == (1, 1, 10.0, 5.0, 2.0, 1, 4, 3, 1.5, 1)
    items = loadfmt.ep_percentages({0: (0.0, {5: 0.0}), 1: (0.0, {})},
                                   {0: (1.0, {5: 0.5}), 1: (1.0, {5: 0.25})}, 2)
    assert items == [(5, 37.5)], items
    assert loadfmt.encode_sumdetail(2, [items]) == struct.pack('<iihhB', 1, 2, 1, 5, 94)
    assert loadfmt.encode_sumdetail(2, []) == struct.pack('<i', 0)
    assert 'ENTRY CHARE 9 "A.b" 0 0' in loadfmt.sts_text({9: 'A.b'})


def gather(proxy):
    local = charm.groups[proxy.gid]
    local.last = None
    proxy.report()
    while local.last is None:
        charm.sleep(0.01)
    return local.last


def main(args):
    test_format()
    proxy = appspector.serve()
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

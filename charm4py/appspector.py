"""Live load data for appSpector (or any CCS client), opt in.

    from charm4py import charm, appspector   # at top level, in every process

    def main(args):
        appspector.serve()        # once, on PE 0, e.g. first thing in main
        ...

    charm.start(main)

Run the job with ``++server ++server-port N`` and point appSpector's viewer
at it: its bottom panes then show, every second, each PE's utilization
stacked by the chares that ran on it (by PE, PE grid) and the utilization by
Python entry method over time.

How it measures. serve() creates a group whose constructor, on every PE,
wraps the few functions through which the Charm++ runtime hands work to
Python (charm.recvArrayMsg, recvGroupMsg, recvChareMsg, recvArrayBcast,
invokeEntryMethod, triggerCallable, and the coroutine resume/pause of the
thread manager). Each wrapped call is timed with time.perf_counter() and its
exclusive time is charged to the chare it ran on and to its entry method;
time a coroutine runs after a resume is charged to the chare and entry
method it was started in. Busy time of a PE is the time spent inside these
calls (message unpickling included), idle time is the rest of the wall
time; time in C++ outside them (reductions, the scheduler) counts as idle.
Nothing is sent anywhere until a client asks: a request to PE 0 broadcasts
to the group, every PE contributes its cumulative counters and a gather
reduction brings them back to PE 0, which replies. The group's own work
(its entry methods and its gather timer) is not counted. Without serve()
nothing is wrapped and nothing costs anything.

Calling serve(). It must be called on PE 0 (anywhere else it raises
Charm4PyError). Calling it again in the same process, also after the
module is reloaded, does nothing and returns the same proxy. The functions
are wrapped once per process: a second AppSpectorLoads group (created by
hand) refuses to construct with Charm4PyError instead of timing every call
twice.

Requests. A gather that has not come back after GATHER_TIMEOUT seconds (a
PE that does not answer) is given up: the requests waiting on it are
answered from the newest gather PE 0 has (ObjLoads), or with the "nothing
yet" reply (SumDetail, or ObjLoads with no gather yet), and the next
request starts a new gather. A gather that arrives late updates the data
but does not answer those requests again. A SumDetail request is answered
from the gather an ObjLoads request just made only if that gather is less
than FRESH_SECONDS old; otherwise it makes its own.

What is not attributed, or attributed coarsely:

  - The mainchare's constructor runs in buildMainchare, which is not
    wrapped, and serve() is called from it: its time before serve()
    returns is idle.
  - A migrated array element starts again from 0 on its new PE (the time
    it ran on the old PE is not carried along); unpickling it and its
    migrated() method are charged to it under the entry method of the
    constructor (the runtime delivers it as a constructor message).
  - Entry methods that charm4py invokes from inside another chare's call
    (a when-condition's queue flushed by a different message, ray futures
    released by the object store, a group handing a message on to local
    objects) are charged to the chare and entry method of that outer call;
    only broadcasts and section deliveries (recvArrayBcast, the section
    manager group) are split per receiving chare.
  - ('g', gid) and ('c', cid) counters are kept while the group or chare
    exists on the PE (charm.groups, charm.chares). charm4py never removes
    groups or chares from those tables, so the counters are bounded by the
    runtime's own tables and are not capped: dropping one would make a
    client that takes differences of replies see a bogus jump.

CCS handlers on PE 0 (reply formats in charm4py.loadfmt, the same as
Charm++'s trace-utilization handlers, so clients decode them unchanged):

    "charm4py ObjLoads"         per PE: wall time, idle time, and the
                                cumulative time of every chare on it
    "charm4py SumDetail"        mean utilization per entry method since
                                the previous such request
    "charm4py SumDetail sts"    entry-method names, .sts text
"""

import time

from greenlet import getcurrent

from .charm import charm, Chare, register
from . import loadfmt

OBJLOADS_HANDLER = "charm4py ObjLoads"
SUMDETAIL_HANDLER = "charm4py SumDetail"
STS_HANDLER = "charm4py SumDetail sts"

GATHER_TIMEOUT = 5.0    # seconds PE 0 waits for a gather before giving up on it
FRESH_SECONDS = 1.5     # SumDetail may reuse an ObjLoads gather younger than this

_clock = time.perf_counter


def serve():
    """Start measuring on every PE and answer load requests on PE 0. Call
    on PE 0 after charm.start (from main); on another PE it raises
    Charm4PyError. From a coroutine (main is one) it returns once every PE
    measures and the handlers are registered. Returns the group's proxy;
    a second call returns the same proxy and does nothing else."""
    from .charm import Charm4PyError
    if charm.myPe() != 0:
        raise Charm4PyError('appspector.serve() must be called on PE 0, not on PE %d'
                            % charm.myPe())
    # kept on the charm object, not in this module, so that a reloaded
    # module does not create a second group
    proxy = getattr(charm, '_appspector_proxy', None)
    if proxy is None:
        from .chare import Group
        proxy = charm._appspector_proxy = Group(AppSpectorLoads)
        if hasattr(proxy, 'creation_future'):
            charm.awaitCreation(proxy)
    return proxy


def _objkey(obj):
    """Internal key of the chare obj: ('a', aid, index), ('g', gid) or
    ('c', cid)."""
    p = obj.thisProxy
    aid = getattr(p, 'aid', None)
    if aid is not None:
        return ('a', aid, obj.thisIndex)
    gid = getattr(p, 'gid', None)
    if gid is not None:
        return ('g', gid)
    return ('c', getattr(p, 'cid', None))


class Tracker(object):
    """Per-PE accounting: exclusive seconds per chare and per entry method,
    busy seconds of the PE, from a stack of timed calls."""

    def __init__(self, main_gr, skip_gids=(), fanout_gids=()):
        self.start = _clock()
        self.busy = 0.0
        self.objs = {}      # internal key -> seconds
        self.eps = {}       # ep index -> seconds
        # frames: [key, ep, t0, child seconds, greenlet, dropped]
        self.stack = []
        self.main_gr = main_gr
        self.skip = frozenset(skip_gids)        # groups not to measure (our own)
        self.fanout = frozenset(fanout_gids)    # groups that deliver to other chares
        self.originals = {}
        self.quiet = None   # callables of this object run by triggerCallable are not measured

    def _drop_stale(self):
        # On the main greenlet every open frame was opened on it, unless a
        # coroutine paused inside a frame it opened itself: such a frame is
        # dropped (its time is lost) so the PE's busy time stays right.
        st = self.stack
        main = self.main_gr
        while st and st[-1][4] is not main:
            st.pop()[5] = True

    def run(self, key, ep, fn, args):
        st = self.stack
        gr = getcurrent()
        if gr is self.main_gr and st and st[-1][4] is not gr:
            self._drop_stale()
        fr = [key, ep, 0.0, 0.0, gr, False]
        st.append(fr)
        fr[2] = t0 = _clock()
        try:
            return fn(*args)
        finally:
            dt = _clock() - t0
            if not fr[5]:
                if st and st[-1] is fr:
                    st.pop()
                else:
                    try:
                        st.remove(fr)
                    except ValueError:
                        pass
                    if getcurrent() is self.main_gr:
                        self._drop_stale()
                if st:
                    st[-1][3] += dt
                else:
                    self.busy += dt
                ex = dt - fr[3]
                if key is not None:
                    self.objs[key] = self.objs.get(key, 0.0) + ex
                if ep is not None:
                    self.eps[ep] = self.eps.get(ep, 0.0) + ex

    def install(self, c, tm):
        """Wrap the dispatch functions of charm c and thread manager tm (as
        instance attributes, so the Cython layer, which looks them up on
        every message, calls the wrappers). Once per process: if c is
        already wrapped it raises Charm4PyError."""
        if getattr(c, '_appspector_tracker', None) is not None:
            from .charm import Charm4PyError
            raise Charm4PyError('charm4py.appspector is already measuring in this process '
                                '(use appspector.serve(), once, on PE 0)')
        c._appspector_tracker = self
        run, st, skip, fanout = self.run, self.stack, self.skip, self.fanout
        o = self.originals
        for name in ('recvArrayMsg', 'recvGroupMsg', 'recvChareMsg', 'recvArrayBcast',
                     'invokeEntryMethod', 'triggerCallable'):
            o[name] = getattr(c, name)
        for name in ('resumeThread', 'pauseThread'):
            o[name] = getattr(tm, name)

        def recvArrayMsg(aid, index, ep, msg, dcopy_start, _o=o['recvArrayMsg']):
            return run(('a', aid, index), ep, _o, (aid, index, ep, msg, dcopy_start))

        def recvGroupMsg(gid, ep, msg, dcopy_start, _o=o['recvGroupMsg']):
            if gid in skip:
                return _o(gid, ep, msg, dcopy_start)
            return run(None if gid in fanout else ('g', gid), ep, _o, (gid, ep, msg, dcopy_start))

        def recvChareMsg(chare_id, ep, msg, dcopy_start, _o=o['recvChareMsg']):
            return run(('c', chare_id), ep, _o, (chare_id, ep, msg, dcopy_start))

        def recvArrayBcast(aid, indexes, ep, msg, dcopy_start, _o=o['recvArrayBcast']):
            # unpickling is charged to the entry method only; each element's
            # call is charged to it by invokeEntryMethod below
            return run(None, ep, _o, (aid, indexes, ep, msg, dcopy_start))

        def invokeEntryMethod(obj, ep, header, args, ret_fut=False, _o=o['invokeEntryMethod']):
            if st and st[-1][0] is None:        # a broadcast or section delivery
                return run(_objkey(obj), ep, _o, (obj, ep, header, args, ret_fut))
            return _o(obj, ep, header, args, ret_fut)

        callables = getattr(c, 'sched_callables', {})

        def triggerCallable(tag, _o=o['triggerCallable']):
            quiet = self.quiet
            if quiet is not None:
                cb = callables.get(tag)
                if cb is not None and getattr(cb[0], '__self__', None) is quiet:
                    return _o(tag)
            return run(None, loadfmt.TIMER_EP, _o, (tag,))

        def resumeThread(gr, arg, _o=o['resumeThread']):
            where = getattr(gr, '_appspector', None)
            if where is None:
                obj = getattr(gr, 'obj', None)
                where = (_objkey(obj) if obj is not None else None, loadfmt.RESUME_EP)
            return run(where[0], where[1], _o, (gr, arg))

        def pauseThread(_o=o['pauseThread']):
            if st:
                fr = st[-1]
                getcurrent()._appspector = (fr[0], fr[1])
            return _o()

        for name, fn in (('recvArrayMsg', recvArrayMsg), ('recvGroupMsg', recvGroupMsg),
                         ('recvChareMsg', recvChareMsg), ('recvArrayBcast', recvArrayBcast),
                         ('invokeEntryMethod', invokeEntryMethod),
                         ('triggerCallable', triggerCallable)):
            setattr(c, name, fn)
        tm.resumeThread = resumeThread
        tm.pauseThread = pauseThread

    def snapshot(self, pe, arrays, groups, chares):
        """(pe, wall now, elapsed, idle, objs, eps) with objs as
        loadfmt.encode_objloads wants them. arrays, groups, chares: the
        chares that exist on this PE (charm.arrays, charm.groups,
        charm.chares); the counters of the others (array elements migrated
        away or deleted, groups and chares that are gone) are forgotten."""
        elapsed = _clock() - self.start
        objs = []
        for key, secs in list(self.objs.items()):
            kind = key[0]
            if kind == 'a':
                here = arrays.get(key[1])
                if here is None or key[2] not in here:
                    del self.objs[key]
                    continue
                objs.append((key[1], loadfmt.obj_id(key[2]), secs, True))
            elif kind == 'g':
                if key[1] not in groups:
                    del self.objs[key]
                    continue
                objs.append((key[1], pe, secs, False))
            else:
                cid = key[1]
                if cid is None or cid not in chares:
                    del self.objs[key]
                    continue
                objs.append((-1, loadfmt.chare_obj_id(cid), secs, False))
        idle = max(0.0, elapsed - self.busy)
        return (pe, time.time(), elapsed, idle, objs, dict(self.eps))


class Requests(object):
    """PE 0: the CCS requests and the gathers that answer them. No charm
    calls of its own (they are passed in), so it can be tested alone.

    start_gather(gen) asks every PE for a snapshot; gathered(items) must
    then be called with the gather, items = [(gen, snapshot), ...].
    delay_reply() returns a token, send_delayed(token, bytes) and
    send_reply(bytes) answer, schedule(fn, secs, args) runs fn(*args) later.
    """

    def __init__(self, num_pes, start_gather, delay_reply, send_delayed, send_reply,
                 schedule, clock=_clock, timeout=GATHER_TIMEOUT, fresh=FRESH_SECONDS):
        self.num_pes = num_pes
        self.start_gather = start_gather
        self.delay_reply = delay_reply
        self.send_delayed = send_delayed
        self.send_reply = send_reply
        self.schedule = schedule
        self.clock = clock
        self.timeout = timeout
        self.fresh = fresh
        self.gen = 0             # number of the newest gather started
        self.gathering = False   # gather self.gen is outstanding
        self.waiting = []        # (kind, token) until it is back or timed out
        self.last = None         # the newest gather, one snapshot per PE
        self.last_t = None       # when it arrived (clock())
        self.ep_prev = None      # {pe: (now, eps)} at the last SumDetail reply
        self.ep_fresh = False    # self.last has not been used for SumDetail yet

    def request(self, kind):
        """kind 'obj' or 'ep': answer once a gather started now is back."""
        self.waiting.append((kind, self.delay_reply()))
        if not self.gathering:
            self.gathering = True
            self.gen += 1
            gen = self.gen
            self.schedule(self.timed_out, self.timeout, [gen])
            self.start_gather(gen)

    def sumdetail(self):
        # A viewer asks for objects and then for entry methods each second:
        # the second request is answered from the gather the first one made,
        # if that is recent.
        if self.ep_fresh and self.clock() - self.last_t < self.fresh:
            self.send_reply(self.reply('ep'))
        else:
            self.request('ep')

    def gathered(self, items):
        # Reductions arrive in order and each PE contributes to them in
        # order, so every gather is newer, PE by PE, than the one before,
        # even one that comes after its timeout. Broadcasts may overtake
        # each other: a gather answers the outstanding requests only if
        # every PE made its snapshot for the newest gather (or later).
        self.last = [snap for _, snap in items]
        self.last_t = self.clock()
        self.ep_fresh = True
        gen = min(g for g, _ in items) if items else 0
        if self.gathering and gen >= self.gen:
            self.gathering = False
            self._answer(False)

    def timed_out(self, gen):
        if self.gathering and gen == self.gen:
            self.gathering = False
            self._answer(True)

    def _answer(self, timed_out):
        waiting, self.waiting = self.waiting, []
        ep_msg = None
        for kind, token in waiting:
            if kind == 'obj':
                msg = self.reply('obj')
            else:
                # every SumDetail request of this gather gets the same interval
                if ep_msg is None:
                    if timed_out and not self.ep_fresh:
                        ep_msg = loadfmt.encode_sumdetail(self.num_pes, [])
                    else:
                        ep_msg = self.reply('ep')
                msg = ep_msg
            self.send_delayed(token, msg)

    def reply(self, kind):
        if kind == 'obj':
            return loadfmt.encode_objloads(p[:5] for p in (self.last or ()))
        if self.last is None:
            return loadfmt.encode_sumdetail(self.num_pes, [])
        cur = {p[0]: (p[1], p[5]) for p in self.last}
        prev, self.ep_prev, self.ep_fresh = self.ep_prev, cur, False
        if prev is None:
            return loadfmt.encode_sumdetail(self.num_pes, [])
        return loadfmt.encode_sumdetail(self.num_pes,
                                        [loadfmt.ep_percentages(prev, cur, self.num_pes)])


@register
class AppSpectorLoads(Chare):

    def __init__(self):
        from .sections import SectionManager
        mygid = self.thisProxy.gid
        fanout = [gid for gid, g in charm.groups.items() if isinstance(g, SectionManager)]
        self.tracker = Tracker(charm.threadMgr.main_gr, skip_gids=[mygid], fanout_gids=fanout)
        self.tracker.install(charm, charm.threadMgr)
        self.requests = None
        if charm.myPe() == 0:
            self.requests = Requests(charm.numPes(), self._start_gather, charm.CcsDelayReply,
                                     charm.CcsSendDelayedReply, charm.CcsSendReply,
                                     charm.scheduleCallableAfter)
            self.tracker.quiet = self.requests
            charm.CcsRegisterHandler(OBJLOADS_HANDLER, self._objloads_handler)
            charm.CcsRegisterHandler(SUMDETAIL_HANDLER, self._sumdetail_handler)
            charm.CcsRegisterHandler(STS_HANDLER, self._sts_handler)

    # -- every PE ----------------------------------------------------------

    def report(self, gen=0):
        snap = self.tracker.snapshot(charm.myPe(), charm.arrays, charm.groups, charm.chares)
        self.reduce(self.thisProxy[0].gathered, (gen, snap), charm.reducers.gather)

    # -- PE 0 --------------------------------------------------------------

    def _start_gather(self, gen):
        self.thisProxy.report(gen)

    def gathered(self, items):
        self.requests.gathered(items)

    def _objloads_handler(self, data):
        self.requests.request('obj')

    def _sumdetail_handler(self, data):
        self.requests.sumdetail()

    def _sts_handler(self, data):
        names = {}
        for ep, em in charm.entryMethods.items():
            names[ep] = '%s.%s' % (em.C.__name__, em.name)
        charm.CcsSendReply(loadfmt.sts_text(names).encode())

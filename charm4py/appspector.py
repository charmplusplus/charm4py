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
reduction brings them back to PE 0, which replies. The group's own work is
not counted. Without serve() nothing is wrapped and nothing costs anything.

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

_clock = time.perf_counter
_group = None


def serve():
    """Start measuring on every PE and answer load requests on PE 0. Call
    once, on one PE, after charm.start (from main). From a coroutine (main
    is one) it returns once every PE measures and the handlers are
    registered. Returns the group's proxy."""
    global _group
    if _group is None:
        from .chare import Group
        _group = Group(AppSpectorLoads)
        if hasattr(_group, 'creation_future'):
            charm.awaitCreation(_group)
    return _group


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
        every message, calls the wrappers)."""
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

        def triggerCallable(tag, _o=o['triggerCallable']):
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

    def snapshot(self, pe, arrays):
        """(pe, wall now, elapsed, idle, objs, eps) with objs as
        loadfmt.encode_objloads wants them. Array elements that are no
        longer on this PE (migrated, deleted) are forgotten."""
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
                objs.append((key[1], pe, secs, False))
            else:
                cid = key[1]
                objs.append((-1, loadfmt.chare_obj_id(cid) if cid else 0, secs, False))
        idle = max(0.0, elapsed - self.busy)
        return (pe, time.time(), elapsed, idle, objs, dict(self.eps))


@register
class AppSpectorLoads(Chare):

    def __init__(self):
        from .sections import SectionManager
        mygid = self.thisProxy.gid
        fanout = [gid for gid, g in charm.groups.items() if isinstance(g, SectionManager)]
        self.tracker = Tracker(charm.threadMgr.main_gr, skip_gids=[mygid], fanout_gids=fanout)
        self.tracker.install(charm, charm.threadMgr)
        self.waiting = []        # PE 0: (kind, delayed reply) until the gather is back
        self.gathering = False
        self.last = None         # PE 0: the newest gather, one tuple per PE
        self.ep_prev = None      # PE 0: {pe: (now, eps)} at the last SumDetail reply
        self.ep_fresh = False    # a gather arrived since the last SumDetail reply
        if charm.myPe() == 0:
            charm.CcsRegisterHandler(OBJLOADS_HANDLER, self._objloads_handler)
            charm.CcsRegisterHandler(SUMDETAIL_HANDLER, self._sumdetail_handler)
            charm.CcsRegisterHandler(STS_HANDLER, self._sts_handler)

    # -- every PE ----------------------------------------------------------

    def report(self):
        self.reduce(self.thisProxy[0].gathered,
                    self.tracker.snapshot(charm.myPe(), charm.arrays), charm.reducers.gather)

    # -- PE 0 --------------------------------------------------------------

    def _request(self, kind):
        self.waiting.append((kind, charm.CcsDelayReply()))
        if not self.gathering:
            self.gathering = True
            self.thisProxy.report()

    def gathered(self, pes):
        self.gathering = False
        self.last = pes
        self.ep_fresh = True
        waiting, self.waiting = self.waiting, []
        for kind, token in waiting:
            charm.CcsSendDelayedReply(token, self._reply(kind))

    def _reply(self, kind):
        if kind == 'obj':
            return loadfmt.encode_objloads(p[:5] for p in self.last)
        cur = {p[0]: (p[1], p[5]) for p in self.last}
        prev, self.ep_prev, self.ep_fresh = self.ep_prev, cur, False
        if prev is None:
            return loadfmt.encode_sumdetail(charm.numPes(), [])
        return loadfmt.encode_sumdetail(charm.numPes(),
                                        [loadfmt.ep_percentages(prev, cur, charm.numPes())])

    def _objloads_handler(self, data):
        self._request('obj')

    def _sumdetail_handler(self, data):
        # A viewer asks for objects and then for entry methods each second:
        # the second request is answered from the gather the first one made.
        if self.ep_fresh:
            charm.CcsSendReply(self._reply('ep'))
        else:
            self._request('ep')

    def _sts_handler(self, data):
        names = {}
        for ep, em in charm.entryMethods.items():
            names[ep] = '%s.%s' % (em.C.__name__, em.name)
        charm.CcsSendReply(loadfmt.sts_text(names).encode())

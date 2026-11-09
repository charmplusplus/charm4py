from . import wait
from . import _ckdyn
from time import time
import sys
from greenlet import getcurrent
_spawn = _ckdyn.spawn


class EntryMethod(object):

    def __init__(self, C, name, profile=False):
        self.C = C  # chare class to which this method belongs to
        self.name = name  # entry method name
        self.epIdx = -1  # entry method index assigned by Charm
        if profile:
            # (time inside entry method, py send overhead, py recv overhead)
            self.times = [0.0, 0.0, 0.0]
            self.running = False

        method = getattr(C, name)
        if hasattr(method, '_ck_coro'):
            if not profile:
                self.run = self._run_th
            else:
                self.run = self._run_th_prof
            self.thread_notify = hasattr(method, '_ck_coro_notify') and method._ck_coro_notify
        else:
            if not profile:
                self.run = self._run
            else:
                self.run = self._run_prof

        self.when_cond = None
        if hasattr(method, 'when_cond'):
            # template object specifying the 'when' condition clause
            # for this entry method
            self.when_cond = getattr(method, 'when_cond')
            if isinstance(self.when_cond, wait.ChareStateMsgCond):
                self.when_cond_func = self.when_cond.cond_func

    def _run(self, obj, header, args, ret_fut=False):
        """ run entry method of the given object in the current thread """
        # set last entry method executed (note that 'last_em_exec' won't
        # necessarily always coincide with the currently running entry method)
        charm.last_em_exec = self
        charm._last_ep_mark = _ckdyn.last_ep()  # what last_ep() was when last_em_exec was set
        try:
            #print(args)
            if ret_fut:
                fut = args[-1]
                args = args[:-1]
            ret = getattr(obj, self.name)(*args)
            if ret_fut and not (ret is None):
                fut.create_object(ret)
        except SystemExit as e:
            charm.exit(exit_code(e))
        except Exception as e:
            charm.process_em_exc(e, obj, header)
            return
        if b'block' in header:
            blockFuture = header[b'block']
            if b'bcast' in header:
                sid = None
                if b'sid' in header:
                    sid = header[b'sid']
                if b'bcastret' in header:
                    charm.contribute(ret, charm.reducers.gather, blockFuture, obj, sid)
                else:
                    charm.contribute(None, None, blockFuture, obj, sid)
            else:
                blockFuture.send(ret)  # send result back to remote

    def _run_prof(self, obj, header, args, ret_fut=False):
        ems = getcurrent().em_callstack
        if len(ems) > 0:
            ems[-1].stopMeasuringTime()
        self.startMeasuringTime()
        ems.append(self)
        exception = None
        try:
            self._run(obj, header, args, ret_fut=ret_fut)
        except Exception as e:
            exception = e
        assert self == ems[-1]
        self.stopMeasuringTime()
        ems.pop()
        if len(ems) > 0:
            ems[-1].startMeasuringTime()
        if exception is not None:
            raise exception

    def _run_coro(self, obj, header, args, ret_fut):
        """ the task a pooled worker greenlet runs for a coroutine entry method """
        gr = getcurrent()
        gr.obj = obj
        gr.notify = self.thread_notify
        try:
            self._run(obj, header, args, ret_fut)
        finally:
            # the pooled greenlet is reused: do not keep obj alive
            gr.obj = None
            gr.notify = None

    def _run_coro_prof(self, obj, header, args, ret_fut):
        gr = getcurrent()
        gr.obj = obj
        gr.notify = self.thread_notify
        gr.em_callstack = [self]
        try:
            self._run(obj, header, args, ret_fut)
        finally:
            gr.obj = None
            gr.notify = None

    def _run_th(self, obj, header, args, ret_fut=False):
        """ run entry method of the given object on a worker greenlet (_ckdyn.spawn).
            The C side brackets every switch into the worker with
            CkDynCoroResume/CkDynCoroSuspend for obj and this entry. """
        obj._numthreads += 1
        try:
            done = _spawn(self.epIdx, self._run_coro, obj, header, args, ret_fut)
        except BaseException:
            obj._numthreads -= 1
            raise
        if done:
            obj._numthreads -= 1

    def _run_th_prof(self, obj, header, args, ret_fut=False):
        ems = getcurrent().em_callstack
        if len(ems) > 0:
            ems[-1].stopMeasuringTime()
        self.startMeasuringTime()
        obj._numthreads += 1
        exception = None
        try:
            done = _ckdyn.spawn(self.epIdx, self._run_coro_prof, obj, header, args, ret_fut)
        except Exception as e:
            exception = e
            done = True
        if done:
            _thread_finished(obj)
        self.stopMeasuringTime()
        if len(ems) > 0:
            ems[-1].startMeasuringTime()
        if exception is not None:
            raise exception

    def startMeasuringTime(self):
        if charm._precvtime > 0:
            self.addRecvTime(time() - charm._precvtime)
            charm._precvtime = -1
        assert not self.running and charm.runningEntryMethod is None
        charm.runningEntryMethod = self
        self.running = True
        self.startTime = time()
        self.sendTime = 0.0
        self.measuringSendTime = False

    def stopMeasuringTime(self):
        assert self.running and charm.runningEntryMethod == self
        self.running = False
        charm.runningEntryMethod = None
        total = time() - self.startTime
        self.times[0] += total - self.sendTime
        self.times[1] += self.sendTime

    def startMeasuringSendTime(self):
        assert not self.measuringSendTime
        self.sendStartTime = time()
        self.measuringSendTime = True

    def stopMeasuringSendTime(self):
        assert self.measuringSendTime
        self.sendTime += time() - self.sendStartTime
        self.measuringSendTime = False

    def addRecvTime(self, t):
        self.times[2] += t

    def __getstate__(self):
        return self.epIdx

    def __setstate__(self, ep):
        self.__dict__.update(charm.entryMethods[ep].__dict__)


# This decorator sets a 'when' condition for the chosen entry method 'func'.
# It is used so that the entry method is invoked only when the condition is true.
# Entry method is guaranteed to be invoked (for any message order) as long as there
# are messages satisfying the condition if AUTO_FLUSH_WAIT_QUEUES = True. Otherwise user
# must manually call chare.__flush_wait_queues__() when the condition becomes true
def when(cond_str):
    def _when(func):
        method_args = {}
        for i in range(1, func.__code__.co_argcount):
            method_args[func.__code__.co_varnames[i]] = i-1
        func.when_cond = wait.parse_cond_str(cond_str, func.__module__, method_args)
        return func
    return _when


def exit_code(e):
    """ the process exit code for a SystemExit raised by an entry method """
    code = e.code
    if code is None:
        code = 0
    if not isinstance(code, int):
        print(code)
        code = 1
    return code


def _thread_finished(obj):
    """ a coroutine of obj finished """
    obj._numthreads -= 1


def entry(**attrs):
    """ Entry-method attributes, given as keyword arguments set to True:
        expedited, inline, immediate, notrace, exclusive, appwork (the meaning of
        each is in the user documentation, docs/chares.rst). Recorded on the
        function and checked when its class is registered (entry_flags). """
    def _entry(func):
        func._ck_entry_attrs = dict(attrs)
        return func
    return _entry


# attribute of @entry -> CkDyn entry flag (ckdyn.h CKDYN_EP_*)
ENTRY_ATTR_FLAGS = {
    'inline':    getattr(_ckdyn, 'EP_INLINE', 1 << 1),
    'expedited': getattr(_ckdyn, 'EP_EXPEDITED', 1 << 2),
    'immediate': getattr(_ckdyn, 'EP_IMMEDIATE', 1 << 3),
    'notrace':   getattr(_ckdyn, 'EP_NOTRACE', 1 << 5),
    'appwork':   getattr(_ckdyn, 'EP_APPWORK', 1 << 6),
    'exclusive': getattr(_ckdyn, 'EP_EXCLUSIVE', 1 << 7),
}
NODEGROUP_ONLY_ATTRS = ('immediate', 'exclusive')


def entry_attrs(func):
    """ the @entry attributes of func, looked up along its __wrapped__ chain """
    while func is not None:
        attrs = getattr(func, '_ck_entry_attrs', None)
        if attrs is not None:
            return attrs
        func = getattr(func, '__wrapped__', None)
    return {}


def entry_flags(C, name, nodegroup):
    """ the CKDYN_EP_* flags of the @entry attributes of method name of C, registered
        for a nodegroup if nodegroup is True. Raises Charm4PyError for an unknown
        attribute, inline on a coroutine, and immediate or exclusive outside a nodegroup. """
    from .charm import Charm4PyError
    func = getattr(C, name, None)
    attrs = entry_attrs(func)
    flags = 0
    for a, v in attrs.items():
        if a not in ENTRY_ATTR_FLAGS:
            raise Charm4PyError('@entry on %s.%s: unknown attribute %r (known: %s)' %
                                (C.__name__, name, a, ', '.join(sorted(ENTRY_ATTR_FLAGS))))
        if not v:
            continue
        if a in NODEGROUP_ONLY_ATTRS and not nodegroup:
            raise Charm4PyError('@entry(%s=True) on %s.%s: allowed only on nodegroup entry methods '
                                '(register the class for NodeGroup only)' % (a, C.__name__, name))
        if a == 'inline' and is_coro(func):
            raise Charm4PyError('@entry(inline=True) on %s.%s: a coroutine entry method cannot be inline'
                                % (C.__name__, name))
        flags |= ENTRY_ATTR_FLAGS[a]
    return flags


def has_nodegroup_only_attrs(C):
    """ whether a method of C has @entry(immediate=True) or @entry(exclusive=True) """
    for m in dir(C):
        f = getattr(C, m, None)
        attrs = entry_attrs(f) if callable(f) else {}
        if any(attrs.get(a) for a in NODEGROUP_ONLY_ATTRS):
            return True
    return False


def is_coro(func):
    """ whether func is a coroutine entry method (@coro), along its __wrapped__ chain """
    while func is not None:
        if getattr(func, '_ck_coro', False):
            return True
        func = getattr(func, '__wrapped__', None)
    return False


def coro(func):
    func._ck_coro = True
    return func


def coro_ext(event_notify=False):
    def _coro(func):
        func._ck_coro = True
        func._ck_coro_notify = event_notify
        return func
    return _coro


def charmStarting():
    global charm
    from .charm import charm


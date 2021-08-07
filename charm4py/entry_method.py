from . import wait
from time import time
import sys
from greenlet import greenlet, getcurrent
from . import charm

class EntryMethodOptions:
    def __init__(self):
        self.value = 0
    def set_option(self, val_identifier):
        self.value |= val_identifier
    def unset_option(self, val_identifier):
        raise NotImplementedError("Options are currently permanent")
    def get(self):
        return self.value

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

        self._msg_opts = None
        if hasattr(method, '_msg_opts'):
            self._msg_opts = method._msg_opts

        self.when_cond = None
        if hasattr(method, 'when_cond'):
            # template object specifying the 'when' condition clause
            # for this entry method
            self.when_cond = getattr(method, 'when_cond')
            if isinstance(self.when_cond, wait.ChareStateMsgCond):
                self.when_cond_func = self.when_cond.cond_func

    def _run(self, obj, header, args):
        """ run entry method of the given object in the current thread """
        # set last entry method executed (note that 'last_em_exec' won't
        # necessarily always coincide with the currently running entry method)
        charm.last_em_exec = self
        try:
            ret = getattr(obj, self.name)(*args)
        except SystemExit:
            exit_code = sys.exc_info()[1].code
            if exit_code is None:
                exit_code = 0
            if not isinstance(exit_code, int):
                print(exit_code)
                exit_code = 1
            charm.exit(exit_code)
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

    def _run_prof(self, obj, header, args):
        ems = getcurrent().em_callstack
        if len(ems) > 0:
            ems[-1].stopMeasuringTime()
        self.startMeasuringTime()
        ems.append(self)
        exception = None
        try:
            self._run(obj, header, args)
        except Exception as e:
            exception = e
        assert self == ems[-1]
        self.stopMeasuringTime()
        ems.pop()
        if len(ems) > 0:
            ems[-1].startMeasuringTime()
        if exception is not None:
            raise exception

    def _run_th(self, obj, header, args):
        gr = greenlet(self._run)
        gr.obj = obj
        gr.notify = self.thread_notify
        obj._numthreads += 1
        gr.switch(obj, header, args)
        if gr.dead:
            obj._numthreads -= 1

    def _run_th_prof(self, obj, header, args):
        ems = getcurrent().em_callstack
        if len(ems) > 0:
            ems[-1].stopMeasuringTime()
        gr = greenlet(self._run)
        gr.obj = obj
        gr.notify = self.thread_notify
        gr.em_callstack = [self]
        self.startMeasuringTime()
        obj._numthreads += 1
        exception = None
        try:
            gr.switch(obj, header, args)
        except Exception as e:
            exception = e
        if gr.dead:
            obj._numthreads -= 1
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


def coro(func):
    func._ck_coro = True
    return func

def expedited(func):
    options = EntryMethodOptions()
    # TODO: get this value from charm
    # options.set_value(charm.em_options.CK_EXPEDITED)
    options.set_option(0x4)
    func._msg_opts = options
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

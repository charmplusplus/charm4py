from . import wait
import time
import sys
from threading import current_thread, get_ident

class EntryMethod(object):

    def __init__(self, C, name, profile=False):
        self.C = C           # chare class to which this method belongs to
        self.name = name     # entry method name
        self.isCtor = False  # true if method is constructor
        self.epIdx = -1      # entry method index assigned by Charm
        self.profile = profile  # true if profiling this entry method's times
        if profile:
            self.times = [0.0, 0.0, 0.0]  # (time inside entry method, py send overhead, py recv overhead)
            self.running = False
            self.run_non_threaded = self._run_non_threaded_prof
        else:
            self.run_non_threaded = self._run_non_threaded

        method = getattr(C, name)

        if hasattr(method, '_ck_threaded'):
            self.isThreaded = True  # true if entry method runs in its own thread
            self.run = self.run_threaded
            self.thread_notify = hasattr(method, '_ck_threaded_notify') and method._ck_threaded_notify
        else:
            self.isThreaded = False
            self.run = self.run_non_threaded

        self.when_cond = None
        if hasattr(method, 'when_cond'):
            # template object specifying the 'when' condition clause for this entry method
            self.when_cond = getattr(method, 'when_cond')
            if isinstance(self.when_cond, wait.ChareStateMsgCond):
                self.when_cond_func = self.when_cond.cond_func

    def run_threaded(self, obj, header, args):
        threadMgr = charm.threadMgr
        if get_ident() == threadMgr.main_thread_id:
            # run entry method of the given object in its own thread
            threadMgr.startThread(obj, self, args, header)
        else:
            # we are already running in a separate thread, use that one
            self.run_non_threaded(obj, header, args)

    def _run_non_threaded(self, obj, header, args):
        """ run entry method of the given object in the current thread """
        charm.last_ntem = self  # last non-threaded entry method
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
                    obj.contribute(ret, charm.reducers.gather, blockFuture, sid)
                else:
                    obj.contribute(None, None, blockFuture, sid)
            else:
                blockFuture.send(ret)  # send result back to remote

    def _run_non_threaded_prof(self, obj, header, args):
        ems = current_thread().em_callstack
        if len(ems) > 0:
            ems[-1].stopMeasuringTime()
        self.startMeasuringTime()
        ems.append(self)
        exception = None
        try:
            self._run_non_threaded(obj, header, args)
        except Exception as e:
            exception = e
        assert self == ems[-1]
        self.stopMeasuringTime()
        ems.pop()
        if len(ems) > 0:
            ems[-1].startMeasuringTime()
        if exception is not None:
            raise exception

    def startMeasuringTime(self):
        if charm._entrytime > 0:
            self.addRecvTime(time.time() - charm._entrytime)
            charm._entrytime = -1
        assert not self.running and charm.runningEntryMethod is None
        charm.runningEntryMethod = self
        self.running = True
        self.startTime = time.time()
        self.sendTime = 0.0
        self.measuringSendTime = False

    def stopMeasuringTime(self):
        assert self.running and charm.runningEntryMethod == self
        self.running = False
        charm.runningEntryMethod = None
        total = time.time() - self.startTime
        self.times[0] += total - self.sendTime
        self.times[1] += self.sendTime

    def startMeasuringSendTime(self):
        assert not self.measuringSendTime
        self.sendStartTime = time.time()
        self.measuringSendTime = True

    def stopMeasuringSendTime(self):
        assert self.measuringSendTime
        self.sendTime += time.time() - self.sendStartTime
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


def threaded(func):
    func._ck_threaded = True
    return func


def threaded_ext(event_notify=False):
    def _threaded(func):
        func._ck_threaded = True
        func._ck_threaded_notify = event_notify
        return func
    return _threaded


charm = None

def charmStarting():
    from .charm import charm
    globals()['charm'] = charm

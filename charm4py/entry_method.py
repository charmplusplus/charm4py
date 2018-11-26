from . import wait
import time
import sys


class EntryMethod(object):

    def __init__(self, C, name, profile=False):
        self.C = C           # chare class to which this method belongs to
        self.name = name     # entry method name
        self.isCtor = False  # true if method is constructor
        self.epIdx = -1      # entry method index assigned by Charm
        self.profile = profile  # true if profiling this entry method's times
        if profile:
            self.times = [0.0, 0.0, 0.0]  # (time inside entry method, py send overhead, py recv overhead)

        method = getattr(C, name)

        if hasattr(method, '_ck_threaded'):
            self.isThreaded = True  # true if entry method runs in its own thread
            self.run = self.run_threaded
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
        """ run entry method of the given object in its own thread """
        charm.threadMgr.startThread(obj, self, args, header)

    def run_non_threaded(self, obj, header, args):
        """ run entry method of the given object in main thread """
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
        if b'block' in header:
            blockFuture = header[b'block']
            if b'bcast' in header:
                obj.contribute(None, None, blockFuture)
            else:
                blockFuture.send(ret)  # send result back to remote

    def startMeasuringTime(self):
        charm.runningEntryMethod = self
        self.startTime = time.time()
        self.sendTime = 0.0
        self.measuringSendTime = False

    def stopMeasuringTime(self):
        charm.runningEntryMethod = None
        self.stopMeasuringSendTime()
        total = time.time() - self.startTime
        self.times[0] += total - self.sendTime
        self.times[1] += self.sendTime

    def startMeasuringSendTime(self):
        self.sendStartTime = time.time()
        self.measuringSendTime = True

    def stopMeasuringSendTime(self):
        if self.measuringSendTime:
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
        func.when_cond = wait.parse_cond_str(cond_str, method_args)
        return func
    return _when


def threaded(func):
    func._ck_threaded = True
    return func


def charmStarting():
    from .charm import charm
    globals()['charm'] = charm

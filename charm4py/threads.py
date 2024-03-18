from greenlet import getcurrent


# Future IDs (fids) are sometimes carried as reference numbers inside
# Charm++ CkCallback objects. The data type most commonly used for
# this is unsigned short, hence this limit
FIDMAXVAL = 65535


class NotThreadedError(Exception):
    def __init__(self, msg):
        super(NotThreadedError, self).__init__(msg)
        self.message = msg


# NOTE: currently objects with active threads cannot migrate (they have to
# finish their threaded entry methods to do so). So, we can assume that
# the result of a non-collective future can be sent to the PE where the future
# was created. And therefore we don't need to include a proxy to the source
# chare when sending a future.
# Also, there is no situation currently where a collective future needs to
# be pickled (so again we don't need to include the proxy).
# See commit 25e2935 if need to resurrect code where proxies were included when
# futures were pickled.

class Future(object):

    def __init__(self, fid, gr, src, num_vals):
        self.fid = fid  # unique future ID within the process that created it
        self.gr = gr  # greenlet that created the future
        self.src = src  # PE where the future was created (not used for collective futures)
        self.nvals = num_vals  # number of values that the future expects to receive
        self.values = []  # values of the future
        self.blocked = False  # flag to check if creator thread is blocked on the future
        self.gotvalues = False  # flag to check if expected number of values have been received
        self.error = None  # if the future receives an Exception, it is set here

    def get(self):
        """ Blocking call on current entry method's thread to obtain the values of the
            future. If the values are already available then they are returned immediately.
        """
        if not self.gotvalues:
            self.blocked = True
            self.gr = getcurrent()
            self.values = threadMgr.pauseThread()

        if self.error is not None:
            raise self.error

        if self.nvals == 1:
            return self.values[0]
        return self.values

    def ready(self):
        return self.gotvalues

    def waitReady(self, f):
        self.blocked = 2

    def send(self, result=None):
        """ Send a value to this future. """
        charm.thisProxy[self.src]._future_deposit_result(self.fid, result)

    def __call__(self, result=None):
        self.send(result)

    def getTargetProxyEntryMethod(self):
        return charm.thisProxy[self.src]._future_deposit_result

    def deposit(self, result):
        """ Deposit a value for this future. """
        self.values.append(result)
        if isinstance(result, Exception):
            self.error = result
        if len(self.values) == self.nvals:
            self.gotvalues = True
            return True
        return False

    def resume(self, threadMgr):
        if self.blocked == 2:
            # someone is waiting for future to become ready, signal by sending myself
            self.blocked = False
            threadMgr.resumeThread(self.gr, self)
        elif self.blocked:
            self.blocked = False
            # someone is waiting on the future, signal by sending the values
            threadMgr.resumeThread(self.gr, self.values)

    def __getstate__(self):
        return (self.fid, self.src)

    def __setstate__(self, state):
        self.fid, self.src = state


class CollectiveFuture(Future):

    def __init__(self, fid, gr, proxy, num_vals):
        super(CollectiveFuture, self).__init__(fid, gr, -1, num_vals)
        self.proxy = proxy

    def getTargetProxyEntryMethod(self):
        return self.proxy._coll_future_deposit_result

    def send(self, result=None):
        self.proxy._coll_future_deposit_result(self.fid, result)


# LocalFuture is a future meant to be used strictly locally. It should not be
# be sent to other PEs. It is more lightweight than a regular future: creation,
# sending to the future and resuming is faster.
class LocalFuture(object):

    def __init__(self):
        self.gr = getcurrent()  # greenlet that created the future

    def send(self, result=None):
        threadMgr.resumeThread(self.gr, result)

    def get(self):
        return threadMgr.pauseThread()


class EntryMethodThreadManager(object):

    def __init__(self, _charm):
        global charm, Charm4PyError, threadMgr
        from .charm import Charm4PyError
        charm = _charm
        threadMgr = self
        self.options = charm.options
        self.lastfid = 0  # future ID of the last future created on this PE
        self.futures = {}  # future ID -> Future object
        self.coll_futures = {}  # (future ID, obj) -> CollectiveFuture object

    def start(self):
        self.main_gr = getcurrent()  # main greenlet
        if not self.options.profiling:
            self.resumeThread = self._resumeThread
        else:
            self.resumeThread = self.resumeThread_prof
            self.main_gr.em_callstack = []

    def isMainThread(self):
        return getcurrent() == self.main_gr

    def objMigrating(self, obj):
        if obj._numthreads > 0:
            raise Charm4PyError('Migration of chares with active threads is not currently supported')

    def throwNotThreadedError(self):
        raise NotThreadedError("Method '" + charm.last_em_exec.C.__name__ + "." +
                               charm.last_em_exec.name +
                               "' must be a couroutine to be able to suspend (decorate it with @coro)")

    def pauseThread(self):
        """ Called by an entry method thread to wait for something.
            Returns data that the thread was waiting for, or None if it was
            waiting for an event
        """
        gr = getcurrent()
        main_gr = self.main_gr
        if gr == main_gr:
            self.throwNotThreadedError()  # verify that not running on main thread
        if gr.notify:
            obj = gr.obj
            obj._thread_notify_target.threadPaused(obj._thread_notify_data)
        if gr.parent != main_gr:
            # this can happen with threaded chare constructors that are called
            # "inline" by Charm++ on the PE where the collection is created.
            # Initially it will switch back to the parent thread, but after that
            # we make the parent to be the main thread
            parent = gr.parent
            gr.parent = main_gr
            return parent.switch()
        else:
            return main_gr.switch()

    def _resumeThread(self, gr, arg):
        """ Deposit a result or signal that a local entry method thread is waiting on,
            and resume it. This executes on the main thread.
        """
        #assert getcurrent() == self.main_gr
        if gr.notify:
            obj = gr.obj
            obj._thread_notify_target.threadResumed(obj._thread_notify_data)
        gr.switch(arg)
        if gr.dead:
            gr.obj._numthreads -= 1

    def resumeThread_prof(self, gr, arg):
        ems = getcurrent().em_callstack
        if len(ems) > 0:
            ems[-1].stopMeasuringTime()
        gr.em_callstack[-1].startMeasuringTime()
        self._resumeThread(gr, arg)
        gr.em_callstack[-1].stopMeasuringTime()
        if len(ems) > 0:
            ems[-1].startMeasuringTime()

    def createFuture(self, num_vals=1):
        """ Creates a new Future object by obtaining a unique (local) future ID. """
        gr = getcurrent()
        if gr == self.main_gr:
            self.throwNotThreadedError()
        # get a unique local Future ID
        global FIDMAXVAL
        futures = self.futures
        assert len(futures) < FIDMAXVAL, 'Too many pending futures, cannot create more'
        fid = (self.lastfid % FIDMAXVAL) + 1
        while fid in futures:
            fid = (fid % FIDMAXVAL) + 1
        self.lastfid = fid
        f = Future(fid, gr, charm._myPe, num_vals)
        futures[fid] = f
        return f

    def createCollectiveFuture(self, fid, obj, proxy):
        """ fid is supplied in this case and has to be the same for all distributed chares """
        gr = getcurrent()
        if gr == self.main_gr:
            self.throwNotThreadedError()
        f = CollectiveFuture(fid, gr, proxy, 1)
        self.coll_futures[(fid, obj)] = f
        return f

    def depositFuture(self, fid, result):
        """ Set a value of a future that is being managed by this ThreadManager. """
        futures = self.futures
        try:
            f = futures[fid]
        except KeyError:
            raise Charm4PyError('No pending future with fid=' + str(fid) + '. A common reason is '
                                'sending to a future that already received its value(s)')
        if f.deposit(result):
            del futures[fid]
            # resume if a thread is blocked on the future
            obj = f.gr.obj
            f.resume(self)
            # this is necessary because the result is being deposited from an
            # entry method of CharmRemote, not the object that we resumed
            if self.options.auto_flush_wait_queues and obj._cond_next is not None:
                obj.__flush_wait_queues__()

    def depositCollectiveFuture(self, fid, result, obj):
        f = self.coll_futures[(fid, obj)]
        if f.deposit(result):
            del self.coll_futures[(fid, obj)]
            del f.proxy
            f.resume(self)

    def cancelFuture(self, f):
        fid = f.fid
        del self.futures[fid]
        f.gotvalues = True
        f.values = [None] * f.nvals
        f.resume(self)

    # TODO: method to cancel collective future. the main issue with this is
    # that the future would need to be canceled on every chare waiting on it

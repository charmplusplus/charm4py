import array
from greenlet import greenlet, getcurrent


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
# futures were pickled

class Future(object):

    def __init__(self, fid, gr, src, nsenders):
        self.fid = fid  # unique future ID within process
        self.gr = gr  # greenlet that created the future
        self.src = src
        self.nsenders = nsenders  # number of senders
        self.values = []  # values of the future (can be multiple in case of multiple senders)
        self.blocked = False  # flag to check if creator thread is blocked on future
        self.gotvalues = False  # flag to check if values have been received

    def get(self):
        """ Blocking call on current entry method's thread to obtain the values of the
            future. If the values are already available then they are returned immediately.
        """
        if not self.gotvalues:
            self.blocked = True
            self.values = charm.threadMgr.pauseThread()

        if self.nsenders == 1:
            val = self.values[0]
            if isinstance(val, Exception):
                raise val
            return val

        for val in self.values:
            if isinstance(val, Exception):
                raise val
        return self.values

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
        if len(self.values) == self.nsenders:
            self.gotvalues = True
            return True
        return False

    def resume(self, threadMgr):
        if self.blocked:
            self.blocked = False
            threadMgr.resumeThread(self.gr, self.values)

    def __getstate__(self):
        return (self.fid, self.src)

    def __setstate__(self, state):
        self.fid, self.src = state


class CollectiveFuture(Future):

    def __init__(self, fid, gr, proxy, nsenders):
        super(CollectiveFuture, self).__init__(fid, gr, -1, nsenders)
        self.proxy = proxy

    def getTargetProxyEntryMethod(self):
        return self.proxy._coll_future_deposit_result

    def send(self, result=None):
        self.proxy._coll_future_deposit_result(self.fid, result)


class EntryMethodThreadManager(object):

    def __init__(self):
        global charm, Charm4PyError
        from .charm import charm, Charm4PyError
        self.options = charm.options
        self.main_gr = getcurrent()  # main greenlet
        # pool of Future IDs for futures created by this ThreadManager. Can
        # have as many active futures as the size of this pool
        self.fidpool = array.array('H', range(30000, 0, -1))
        self.futures = {}  # future ID -> Future object
        self.coll_futures = {}  # (future ID, obj) -> CollectiveFuture object
        if not self.options.profiling:
            self.resumeThread = self._resumeThread
        else:
            self.resumeThread = self.resumeThread_prof
            self.main_gr.em_callstack = []

    def isMainThread(self):
        return getcurrent() == self.main_gr

    def objMigrating(self, obj):
        if obj._numthreads > 0:
            raise Charm4PyError('Migration of chares with active threads is not yet supported')

    def throwNotThreadedError(self):
        raise NotThreadedError("Entry method '" + charm.last_em_exec.C.__name__ + "." +
                               charm.last_em_exec.name +
                               "' must be marked as 'threaded' to block")

    def pauseThread(self):
        """ Called by an entry method thread to wait for something.
            Returns data that the thread was waiting for, or None if it was waiting for an event
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

    def createFuture(self, senders=1):
        """ Creates a new Future object by obtaining/creating a unique future ID. The
            future also has some attributes related to the creator object's proxy to allow
            remote chares to send values to the future's creator. The new future object is
            saved in the self.futures dict, and will be deleted whenever its values are received.
        """
        gr = getcurrent()
        if gr == self.main_gr:
            self.throwNotThreadedError()
        fid = self.fidpool.pop()
        f = Future(fid, gr, charm._myPe, senders)
        self.futures[fid] = f
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
        f = self.futures[fid]
        if f.deposit(result):
            del self.futures[fid]
            self.fidpool.append(fid)
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
        self.fidpool.append(fid)
        f.gotvalues = True
        f.values = [None] * f.nsenders
        f.resume(self)

    # TODO: method to cancel collective future. the main issue with this is
    # that the future would need to be canceled on every chare waiting on it

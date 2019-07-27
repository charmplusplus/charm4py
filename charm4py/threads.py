import sys
import threading
if sys.version_info < (3, 0, 0):
    from thread import get_ident
else:
    from threading import get_ident
import array


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

    def __init__(self, fid, thread_state, src, nsenders):
        self.fid = fid                    # unique future ID within process
        self.thread_state = thread_state  # thread context where the future is created
        self.src = src                    # PE where the future was created
        self.nsenders = nsenders          # number of senders
        self.values = []                  # values of the future (can be multiple in case of multiple senders)
        self.blocked = False              # flag to check if creator thread is blocked on future
        self.gotvalues = False            # flag to check if values have been received

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
            threadMgr.resumeThread(self.thread_state, self.values)

    def __getstate__(self):
        return (self.fid, self.src)

    def __setstate__(self, state):
        self.fid, self.src = state


class CollectiveFuture(Future):

    def __init__(self, fid, tstate, proxy, nsenders):
        super(CollectiveFuture, self).__init__(fid, tstate, -1, nsenders)
        self.proxy = proxy

    def getTargetProxyEntryMethod(self):
        return self.proxy._coll_future_deposit_result

    def send(self, result=None):
        self.proxy._coll_future_deposit_result(self.fid, result)


class ThreadState(object):
    def __init__(self, tid, obj, entry_method):
        self.tid = tid                  # thread ID
        self.obj = obj                  # chare that is using the thread
        self.em = entry_method          # EntryMethod object for which this thread is running
        self.wait_cv = threading.Condition()  # condition variable to pause/resume entry method threads
        self.wait_result = None         # to place data that thread was waiting on
        self.error = None               # to pass exceptions from entry method thread to main thread
        self.idle = False               # True if thread is idle
        self.finished = False           # True if thread has run to completion


class EntryMethodThreadManager(object):
    """ Creates and manages entry method threads """

    def __init__(self):
        self.PROFILING = Options.profiling
        self.main_thread_id = get_ident()    # ID of the charm4py process main thread
        # condition variable used by main thread to pause while threaded entry method is running
        self.entryMethodRunning = threading.Condition()
        self.threads = {}                    # thread ID -> ThreadState object
        # pool of Future IDs for futures created by this ThreadManager. Can have as many
        # active futures as the size of this pool
        self.fidpool = array.array('H', range(30000,0,-1))
        self.futures = {}                    # future ID -> Future object
        self.coll_futures = {}               # (future ID, obj) -> Future object
        self.threadPool = []

    def isMainThread(self):
        return get_ident() == self.main_thread_id

    def startThread(self, obj, entry_method, args, header):
        """ Called by main thread to spawn an entry method thread """

        #assert get_ident() == self.main_thread_id
        if len(self.threadPool) > 0:
            thread_state = self.threadPool.pop()
            self.resumeThread(thread_state, (obj, entry_method, args, header))
        else:
            with self.entryMethodRunning:
                t = threading.Thread(target=self.entryMethodRun_thread,
                                     args=(obj, entry_method, args, header))
                if self.PROFILING:
                    ems = threading.current_thread().em_callstack
                    if len(ems) > 0:
                        ems[-1].stopMeasuringTime()
                t.start()
                self.entryMethodRunning.wait()  # wait until entry method finishes OR pauses
                self.threadStopped(self.threads[t.ident])
                if self.PROFILING:
                    if len(ems) > 0:
                        ems[-1].startMeasuringTime()

    def threadStopped(self, thread_state):
        """ Called by main thread when entry method thread has finished/paused """

        if thread_state.idle:
            self.threadPool.append(thread_state)  # return to thread pool
        elif thread_state.finished:
            del self.threads[thread_state.tid]
        else:
            # thread paused inside entry method, or returned error
            error = thread_state.error
            if error is not None:
                thread_state.obj._num_threads -= 1
                del self.threads[thread_state.tid]
                raise error

    def objMigrating(self, obj):
        if obj._num_threads > 0:
            raise Charm4PyError("Migration of chares with active threads is not yet supported")

    def entryMethodRun_thread(self, obj, entry_method, args, header):
        """ Entry method thread main function """
        tid = get_ident()
        thread_state = ThreadState(tid, obj, entry_method)
        self.threads[tid] = thread_state
        if self.PROFILING:
            ems = [entry_method]
            threading.current_thread().em_callstack = ems
            entry_method.startMeasuringTime()
        with self.entryMethodRunning:
            try:
                while True:
                    thread_state.idle = False
                    obj._num_threads += 1
                    thread_state.notify = entry_method.thread_notify
                    try:
                        ret = getattr(obj, entry_method.name)(*args)  # invoke entry method
                        if b'block' in header:
                            if b'bcast' in header:
                                sid = None
                                if b'sid' in header:
                                    sid = header[b'sid']
                                if b'bcastret' in header:
                                    obj.contribute(ret, charm.reducers.gather, header[b'block'], sid)
                                else:
                                    obj.contribute(None, None, header[b'block'], sid)
                            else:
                                header[b'block'].send(ret)
                    except Exception as e:
                        charm.process_em_exc(e, obj, header)
                    if self.PROFILING:
                        entry_method.stopMeasuringTime()
                        ems.pop()
                        assert len(ems) == 0
                    thread_state.notify = False
                    thread_state.idle = True
                    thread_state.obj = None
                    obj._num_threads -= 1
                    obj, entry_method, args, header = charm.threadMgr.pauseThread()
                    if obj is None:
                        thread_state.finished = True
                        break
                    thread_state.obj = obj
                    thread_state.em = entry_method
                    if self.PROFILING:
                        entry_method.startMeasuringTime()
                        ems.append(entry_method)
            except SystemExit:
                exit_code = sys.exc_info()[1].code
                if exit_code is None:
                    exit_code = 0
                if not isinstance(exit_code, int):
                    print(exit_code)
                    exit_code = 1
                charm.exit(exit_code)
            except Exception:
                thread_state.error = sys.exc_info()[1]  # store exception for main thread
                if self.PROFILING and entry_method.running:
                    entry_method.stopMeasuringTime()
            self.entryMethodRunning.notify()  # notify main thread that done

    def throwNotThreadedError(self):
        raise NotThreadedError("Entry method '" + charm.last_ntem.C.__name__ + "." +
                               charm.last_ntem.name +
                               "' must be marked as 'threaded' to block")

    def pauseThread(self):
        """ Called by an entry method thread to wait for something.
            Returns data that the thread was waiting for, or None if it was waiting for an event
        """
        tid = get_ident()
        if tid == self.main_thread_id:
            self.throwNotThreadedError()  # verify that not running on main thread
        # thread has entryMethodRunning lock already because it's running an entry method
        thread_state = self.threads[tid]
        if thread_state.notify:
            obj = thread_state.obj
            obj._thread_notify_target.threadPaused(obj._thread_notify_data)
        if self.PROFILING:
            ems = threading.current_thread().em_callstack
            if len(ems) > 0:
                ems[-1].stopMeasuringTime()
        with thread_state.wait_cv:
            self.entryMethodRunning.notify()    # notify main thread that I'm pausing
            self.entryMethodRunning.release()
            thread_state.wait_cv.wait()
            self.entryMethodRunning.acquire()   # got what I was waiting for, resuming
            #print("result got here", tid, thread_state.wait_result)
        if self.PROFILING:
            if len(ems) > 0:
                ems[-1].startMeasuringTime()
        #self.entryMethodRunning.acquire()
        return thread_state.wait_result

    def resumeThreadTid(self, tid, arg):
        self.resumeThread(self.threads[tid], arg)

    def resumeThread(self, thread_state, arg):
        """ Deposit a result or signal that a local entry method thread is waiting on,
            and resume it. This executes on main thread.
        """
        #assert get_ident() == self.main_thread_id
        thread_state.wait_result = arg
        if thread_state.notify:
            obj = thread_state.obj
            obj._thread_notify_target.threadResumed(obj._thread_notify_data)
        with thread_state.wait_cv:
            thread_state.wait_cv.notify()
            self.entryMethodRunning.acquire()
        #with self.entryMethodRunning:
        if self.PROFILING:
            ems = threading.current_thread().em_callstack
            if len(ems) > 0:
                ems[-1].stopMeasuringTime()
        self.entryMethodRunning.wait()  # main thread waits because threaded entry method resumes
        if self.PROFILING and len(ems) > 0:
            ems[-1].startMeasuringTime()
        self.threadStopped(thread_state)

    def createFuture(self, senders=1):
        """ Creates a new Future object by obtaining/creating a unique future ID. The
            future also has some attributes related to the creator object's proxy to allow
            remote chares to send values to the future's creator. The new future object is
            saved in the self.futures dict, and will be deleted whenever its values are received.
        """
        tid = get_ident()
        if tid == self.main_thread_id:
            self.throwNotThreadedError()
        fid = self.fidpool.pop()
        thread_state = self.threads[tid]
        f = Future(fid, thread_state, charm._myPe, senders)
        self.futures[fid] = f
        return f

    def createCollectiveFuture(self, fid, obj, proxy):
        """ fid is supplied in this case and has to be the same for all distributed chares """
        tid = get_ident()
        if tid == self.main_thread_id:
            self.throwNotThreadedError()
        thread_state = self.threads[tid]
        f = CollectiveFuture(fid, thread_state, proxy, 1)
        self.coll_futures[(fid, obj)] = f
        return f

    def depositFuture(self, fid, result):
        """ Set a value of a future that is being managed by this ThreadManager. """
        f = self.futures[fid]
        if f.deposit(result):
            del self.futures[fid]
            self.fidpool.append(fid)
            # resume if blocked
            f.resume(self)

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


charm, Options, Charm4PyError = None, None, None

def charmStarting():
    from .charm import charm, Charm4PyError
    globals()['charm'] = charm
    globals()['Options'] = charm.options
    globals()['Charm4PyError'] = Charm4PyError

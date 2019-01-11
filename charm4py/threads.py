import sys
import threading
if sys.version_info < (3, 0, 0):
    from thread import get_ident
else:
    from threading import get_ident
import time


class NotThreadedError(Exception):
    def __init__(self, msg):
        super(NotThreadedError, self).__init__(msg)
        self.message = msg


class Future(object):

    def __init__(self, fid, thread_state, proxy, nsenders):
        self.fid = fid                    # unique future ID within process
        self.thread_state = thread_state  # thread context where the future is created
        self.proxy = proxy
        # TODO? only obtain proxy_state if actually serializing the future?
        self.proxy_class_name = proxy.__class__.__name__
        self.proxy_state = proxy.__getstate__()
        self.num_senders = nsenders       # number of senders
        self.value = []                   # value of the future, it can be a bag of values in case of multiple senders
        self.thread_paused = False        # flag to check if creator thread is blocked on future
        self.value_received = False       # flag to check if value has been received

    def get(self):
        """ Blocking call on current entry method's thread to obtain the value of the
            future. If the value is already available then it is returned immediately.
        """
        if not self.value_received:
            self.thread_paused = True
            self.value = charm.threadMgr.pauseThread()

        if len(self.value) == 1: return self.value[0]

        return self.value

    def send(self, result):
        """ Set the value of a future either from remote or current thread. """
        self.getTargetProxyEntryMethod()(self.fid, result)

    def getTargetProxyEntryMethod(self):
        if not hasattr(self, 'proxy'):
            proxy_class = getattr(charm, self.proxy_class_name)
            proxy = proxy_class.__new__(proxy_class)
            proxy.__setstate__(self.proxy_state)
            return proxy._future_deposit_result
        else:
            return self.proxy._future_deposit_result

    def deposit(self, result):
        """ Deposit a value for this future and resume any blocked threads waiting on this
            future.
        """
        self.value.append(result)
        if len(self.value) == self.num_senders:
            self.value_received = True
            if self.thread_paused:
                self.thread_paused = False
                charm.threadMgr.resumeThread(self.thread_state, self.value)

    def __getstate__(self):
        return (self.fid, self.proxy_class_name, self.proxy_state)

    def __setstate__(self, state):
        self.fid, self.proxy_class_name, self.proxy_state = state


class CollectiveFuture(Future):

    def __init__(self, fid, tid, proxy, nsenders):
        super(CollectiveFuture, self).__init__(fid, tid, proxy, nsenders)

    def getTargetProxyEntryMethod(self):
        if not hasattr(self, 'proxy'):
            proxy_class = getattr(charm, self.proxy_class_name)
            proxy = proxy_class.__new__(proxy_class)
            proxy.__setstate__(self.proxy_state)
            return proxy._coll_future_deposit_result
        else:
            return self.proxy._coll_future_deposit_result


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
        self.PROFILING = Options.PROFILING
        self.main_thread_id = get_ident()    # ID of the charm4py process main thread
        # condition variable used by main thread to pause while threaded entry method is running
        self.entryMethodRunning = threading.Condition()
        self.threads = {}                    # thread ID -> ThreadState object
        self.futures_count = 0               # counter used as IDs for futures created by this ThreadManager
        self.futures = {}                    # future ID -> Future object
        self.coll_futures = {}               # (future ID, obj) -> Future object
        self.threadPool = []

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
                t.start()
                self.entryMethodRunning.wait()  # wait until entry method finishes OR pauses
                self.threadStopped(self.threads[t.ident])

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
                thread_state.obj.num_threads -= 1
                del self.threads[thread_state.tid]
                raise error

    def objMigrating(self, obj):
        if obj.num_threads > 0:
            raise Charm4PyError("Migration of chares with active threads is not yet supported")

    def entryMethodRun_thread(self, obj, entry_method, args, header):
        """ Entry method thread main function """
        tid = get_ident()
        thread_state = ThreadState(tid, obj, entry_method)
        self.threads[tid] = thread_state
        with self.entryMethodRunning:
            try:
                while True:
                    thread_state.idle = False
                    obj.num_threads += 1
                    thread_state.notify = entry_method.thread_notify
                    ret = getattr(obj, entry_method.name)(*args)  # invoke entry method
                    if b'block' in header:
                        if b'bcast' in header:
                            obj.contribute(None, None, header[b'block'])
                        else:
                            header[b'block'].send(ret)
                    thread_state.notify = False
                    thread_state.idle = True
                    thread_state.obj = None
                    obj.num_threads -= 1
                    obj, entry_method, args, header = charm.threadMgr.pauseThread()
                    if obj is None:
                        thread_state.finished = True
                        break
                    thread_state.obj = obj
                    thread_state.em = entry_method
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
            self.entryMethodRunning.notify()  # notify main thread that done

    def throwNotThreadedError(self):
        raise NotThreadedError("Entry method '" + charm.mainThreadEntryMethod.C.__name__ + "." +
                               charm.mainThreadEntryMethod.name +
                               "' must be marked as 'threaded' to block")

    def pauseThread(self):
        """ Called by an entry method thread to wait for something.
            Returns data that thread was waiting for, or None if was waiting for an event
        """
        tid = get_ident()
        if tid == self.main_thread_id:
            self.throwNotThreadedError()  # verify that not running on main thread
        # thread has entryMethodRunning lock already because it's running an entry method
        thread_state = self.threads[tid]
        if thread_state.notify:
            obj = thread_state.obj
            obj._thread_notify_target.threadPaused(obj._thread_notify_data)
        with thread_state.wait_cv:
            self.entryMethodRunning.notify()    # notify main thread that I'm pausing
            self.entryMethodRunning.release()
            thread_state.wait_cv.wait()
            self.entryMethodRunning.acquire()   # got what I was waiting for, resuming
            #print("result got here", tid, thread_state.wait_result)
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
            charm.mainThreadEntryMethod.stopMeasuringTime()
            thread_state.em.startMeasuringTime()
        self.entryMethodRunning.wait()  # main thread waits because threaded entry method resumes
        if self.PROFILING:
            thread_state.em.stopMeasuringTime()
            charm.mainThreadEntryMethod.startMeasuringTime()
        self.threadStopped(thread_state)

    def createFuture(self, senders=1):
        """ Creates a new Future object by obtaining/creating a unique future ID. The
            future also has some attributes related to the creator object's proxy to allow
            remote chares to send values to the future's creator. The new future object is
            saved in the self.futures map, and will be deleted whenever its value is received.
        """
        tid = get_ident()
        if tid == self.main_thread_id:
            self.throwNotThreadedError()
        self.futures_count += 1
        fid = self.futures_count
        thread_state = self.threads[tid]
        obj = thread_state.obj
        if hasattr(obj, 'thisIndex'):
            proxy = obj.thisProxy[obj.thisIndex]
        else:
            proxy = obj.thisProxy

        f = Future(fid, thread_state, proxy, senders)
        self.futures[fid] = f
        return f

    def createCollectiveFuture(self, fid):
        """ fid is supplied in this case and has to be the same for all distributed chares """
        tid = get_ident()
        if tid == self.main_thread_id:
            self.throwNotThreadedError()
        thread_state = self.threads[tid]
        obj = thread_state.obj
        proxy = obj.thisProxy
        f = CollectiveFuture(fid, thread_state, proxy, 1)
        self.coll_futures[(fid, obj)] = f
        return f

    def depositFuture(self, fid, result):
        """ Set a value of a future that is being managed by this ThreadManager. The flag to
            track receipt of all values, tied to this future, is also updated and any thread
            that is blocking on this future's value is resumed.
        """
        future = self.futures[fid]
        future.deposit(result)
        if future.value_received:
            del self.futures[fid]

    def depositCollectiveFuture(self, fid, result, obj):
        future = self.coll_futures[(fid, obj)]
        future.deposit(result)
        if future.value_received:
            del self.coll_futures[(fid, obj)]


charm, Options, Charm4PyError = None, None, None

def charmStarting():
    from .charm import charm, Options, Charm4PyError
    globals()['charm'] = charm
    globals()['Options'] = Options
    globals()['Charm4PyError'] = Charm4PyError

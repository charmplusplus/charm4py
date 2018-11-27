import sys
import threading
if sys.version_info < (3, 0, 0):
    from thread import get_ident
else:
    from threading import get_ident
from collections import defaultdict
import time


class Future(object):

    def __init__(self, fid, tid, proxy_class_name, proxy_state, nsenders):
        self.fid = fid                    # unique future ID within process
        self.tid = tid                    # thread ID where the future is created
        self.proxy_class_name = proxy_class_name  # creator object's proxy class name
        self.proxy_state = proxy_state    # creator object's proxy state
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
        """ Set the value of a future either from remote or current thread.
            NOTE: For correct semantics, only one unique thread can send to a future. Not
            multiple.
        """
        proxy_class = getattr(charm, self.proxy_class_name)
        proxy = proxy_class.__new__(proxy_class)
        proxy.__setstate__(self.proxy_state)
        proxy._future_deposit_result(self.fid, result)

    def deposit(self, result):
        """ Deposit a value for this future and resume any blocked threads waiting on this
            future.
        """
        self.value.append(result)
        if len(self.value) == self.num_senders:
            self.value_received = True
            if self.thread_paused:
                self.thread_paused = False
                charm.threadMgr.resumeThread(self.tid, self.value)

    def __getstate__(self):
        return (self.fid, self.proxy_class_name, self.proxy_state)

    def __setstate__(self, state):
        self.fid, self.proxy_class_name, self.proxy_state = state


class EntryMethodThreadManager(object):
    """ Creates and manages entry method threads """

    class ThreadState(object):
        def __init__(self, obj, entry_method):
            self.obj = obj                  # chare that spawned the thread
            self.em = entry_method          # EntryMethod object for which this thread is running
            self.wait_cv = threading.Condition()  # condition variable to pause/resume entry method threads
            self.wait_result = None         # to place data that thread was waiting on
            self.error = None               # to pass exceptions from entry method thread to main thread
            self.finished = False           # True if entry method has run to completion

    def __init__(self):
        self.PROFILING = Options.PROFILING
        self.main_thread_id = get_ident()    # ID of the charm4py process main thread
        # condition variable used by main thread to pause while threaded entry method is running
        self.entryMethodRunning = threading.Condition()
        self.threads = {}                    # thread ID -> ThreadState object
        self.obj_threads = defaultdict(set)  # stores active thread IDs of chares
        self.futures_count = 0               # counter used as IDs for futures created by this ThreadManager
        self.futures = {}                    # future ID -> Future object

    def startThread(self, obj, entry_method, args, header):
        """ Called by main thread to spawn an entry method thread """

        assert get_ident() == self.main_thread_id       # TODO comment out eventually

        with self.entryMethodRunning:
            t = threading.Thread(target=self.entryMethodRun_thread,
                                 args=(obj, entry_method, args, header))
            t.start()
            self.entryMethodRunning.wait()  # wait until entry method finishes OR pauses
            self.threadStopped(t.ident)

    def threadStopped(self, tid):
        """ Called by main thread when entry method thread has finished/paused """

        thread_state = self.threads[tid]
        if thread_state.finished:
            self.obj_threads[thread_state.obj].remove(tid)
            del self.threads[tid]
        else:
            # thread paused or returned error
            error = thread_state.error
            if error is not None:
                self.obj_threads[thread_state.obj].discard(tid)
                del self.threads[tid]
                raise error

    def objMigrating(self, obj):
        if obj in self.obj_threads:
            if len(self.obj_threads.pop(obj)) > 0:
                raise Charm4PyError("Migration of chares with active threads is not yet supported")

    def entryMethodRun_thread(self, obj, entry_method, args, header):
        """ Entry method thread main function """

        with self.entryMethodRunning:
            tid = get_ident()
            thread_state = EntryMethodThreadManager.ThreadState(obj, entry_method)
            self.threads[tid] = thread_state
            try:
                self.obj_threads[obj].add(tid)
                ret = getattr(obj, entry_method.name)(*args)  # invoke entry method
                if b'block' in header:
                    if b'bcast' in header:
                        obj.contribute(None, None, header[b'block'])
                    else:
                        header[b'block'].send(ret)
                thread_state.finished = True
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
        raise Charm4PyError("Entry method '" + charm.mainThreadEntryMethod.C.__name__ + "." +
                           charm.mainThreadEntryMethod.name +
                           "' must be marked as 'threaded' to block")

    def pauseThread(self):
        """ Called by an entry method thread to wait for something.
            Returns data that thread was waiting for, or None if was waiting for an event
        """
        tid = get_ident()
        if tid == self.main_thread_id: self.throwNotThreadedError()  # verify that not running on main thread
        # thread has entryMethodRunning lock already because it's running an entry method
        thread_state = self.threads[tid]
        with thread_state.wait_cv:
            self.entryMethodRunning.notify()    # notify main thread that I'm pausing
            self.entryMethodRunning.release()
            thread_state.wait_cv.wait()
            self.entryMethodRunning.acquire()   # got what I was waiting for, resuming
            #print("result got here", tid, thread_state.wait_result)
        #self.entryMethodRunning.acquire()
        return thread_state.wait_result

    def resumeThread(self, tid, arg):
        """ Deposit a result or signal that a local entry method thread is waiting on,
            and resume it. This executes on main thread.
        """
        assert get_ident() == self.main_thread_id
        thread_state = self.threads[tid]
        thread_state.wait_result = arg
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
        self.threadStopped(tid)

    def createFuture(self, senders=1):
        """ Creates a new Future object by obtaining/creating a unique future ID. The
            future also has some attributes related to the creator object's proxy to allow
            remote chares to send values to the future's creator. The new future object is
            saved in the self.futures map, and will be deleted whenever its value is received.
        """
        tid = get_ident()
        if tid == self.main_thread_id: self.throwNotThreadedError()
        self.futures_count += 1
        fid = self.futures_count
        obj = self.threads[tid].obj
        if hasattr(obj, 'thisIndex'): proxy = obj.thisProxy[obj.thisIndex]
        else: proxy = obj.thisProxy

        f = Future(fid, tid, proxy.__class__.__name__, proxy.__getstate__(), senders)
        self.futures[fid] = f
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


def charmStarting():
    from .charm import charm, Options, Charm4PyError
    globals()['charm'] = charm
    globals()['Options'] = Options
    globals()['Charm4PyError'] = Charm4PyError

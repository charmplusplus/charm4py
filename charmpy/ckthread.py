import sys
import threading
if sys.version_info < (3, 0, 0):
    from thread import get_ident
else:
    from threading import get_ident
from charmpy import CharmPyError, charm, Options
from collections import defaultdict
import time


class EntryMethodThreadManager(object):
    """ Creates and manages entry method threads """

    class ThreadState(object):
        def __init__(self, obj, entry_method):
            self.obj = obj                  # chare that spawned the thread
            self.em = entry_method          # EntryMethod object for which this thread is running
            self.wait_cv = threading.Condition() # condition variable to pause/resume entry method threads
            self.wait_result = None         # to place data that thread was waiting on
            self.error = None               # to pass exceptions from entry method thread to main thread
            self.finished = False           # True if entry method has run to completion

    def __init__(self):
        self.PROFILING = Options.PROFILING
        self.main_thread_id = get_ident()   # ID of the charmpy process main thread
        self.entryMethodRunning = threading.Condition() # condition variable used by main thread to
                                                        # pause while threaded entry method is running
        self.threads = {}                   # thread ID -> ThreadState object
        self.obj_threads = defaultdict(set) # stores active thread IDs of chares

    def startThread(self, obj, entry_method, args, caller):
        """ Called by main thread to spawn an entry method thread """

        assert get_ident() == self.main_thread_id       # TODO comment out eventually

        with self.entryMethodRunning:
            t = threading.Thread(target=self.entryMethodRun_thread,
                                 args=(obj, entry_method, args, caller))
            t.start()
            tid = t.ident
            self.entryMethodRunning.wait()  # wait until entry method finishes OR pauses
            self.threadStopped(tid)

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
                raise CharmPyError("Migration of chares with active threads is not yet supported")

    def entryMethodRun_thread(self, obj, entry_method, args, caller):
        """ Entry method thread main function """

        with self.entryMethodRunning:
            tid = get_ident()
            thread_state = EntryMethodThreadManager.ThreadState(obj, entry_method)
            self.threads[tid] = thread_state
            try:
                self.obj_threads[obj].add(tid)
                ret = getattr(obj, entry_method.name)(*args)  # invoke entry method
                if caller is not None:
                    proxy, remote_tid = caller
                    assert ret is not None, str(ret) + " " + thread_state.em.name
                    proxy._thread_deposit_result(remote_tid, ret)
                thread_state.finished = True
            except Exception:
                thread_state.error = sys.exc_info()[1] # store exception for main thread
            self.entryMethodRunning.notify() # notify main thread that done

    def throwNotThreadedError(self):
        raise CharmPyError("Entry method '" + charm.currentEntryMethod.C.__name__ + "." +
                           charm.currentEntryMethod.name +
                           "' must be marked as 'threaded' to block")

    def pauseThread(self):
        """ Called by an entry method thread to wait for something.
            Returns data that thread was waiting for, or None if was waiting for an event
        """
        tid = get_ident()
        if tid == self.main_thread_id: self.throwNotThreadedError() # verify that not running on main thread
        # thread has entryMethodRunning lock already because it's running an entry method
        thread_state = self.threads[tid]
        wait_time = 0.0
        with thread_state.wait_cv:
            if self.PROFILING: t0 = time.time()
            self.entryMethodRunning.notify()    # notify main thread that I'm pausing
            self.entryMethodRunning.release()
            thread_state.wait_cv.wait()
            if self.PROFILING: wait_time = time.time() - t0
            self.entryMethodRunning.acquire()   # got what I was waiting for, resuming
            #print("result got here", tid, thread_state.wait_result)
        #self.entryMethodRunning.acquire()
        return thread_state.wait_result, wait_time

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
            charm.currentEntryMethod.stopMeasuringTime()
            thread_state.em.startMeasuringTime()
        self.entryMethodRunning.wait() # main thread waits because threaded entry method resumes
        if self.PROFILING:
            thread_state.em.stopMeasuringTime()
            charm.currentEntryMethod.startMeasuringTime()
        self.threadStopped(tid)

    def getReturnHandle(self):
        """ Get handle in order to be able to send remote results to the calling thread
            in this PE.
            The handle contains the proxy to the chare that owns the thread and the thread ID
            of the waiting thread. Result is delivered to an entry method of the chare
            that owns the thread, so that the CPU load of the thread, once it resumes,
            is counted towards that object.
            NOTE: if chare that is waiting for result migrates, tid is not valid.
            We are not supporting migration of chares with active threads for now.
        """
        tid = get_ident()
        if tid == self.main_thread_id: self.throwNotThreadedError() # verify that not running on main thread
        obj = self.threads[tid].obj
        if hasattr(obj, 'thisIndex'): proxy = obj.thisProxy[obj.thisIndex]
        else: proxy = obj.thisProxy
        return (proxy.__class__.__name__, proxy.__getstate__(), tid)

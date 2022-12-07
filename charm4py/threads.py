from greenlet import getcurrent
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import CancelledError, TimeoutError, InvalidStateError
from concurrent.futures._base import CANCELLED, FINISHED, CANCELLED_AND_NOTIFIED
from gevent import Timeout
from sys import stderr
from dataclasses import dataclass
from typing import Union
from collections.abc import Iterable

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

#@dataclass()
class Future(ConcurrentFuture):

    #fid: int  # unique future ID within the process that created it
    #gr: object  # greenlet that created the future
    # PE where the future was created (not used for collective futures)
    #src: int
    #nvals: int  # number of values that the future expects to receive
    #values: list  # values of the future
    # flag to check if creator thread is blocked on the future
    #blocked: Union[bool, int] = False
    #gotvalues: bool = False  # flag to check if expected number of values have been received
    # if the future receives an Exception, it is set here
    #error: Union[None, Exception] = None
    #callbacks: list  # list of callback functions
    #is_cancelled: bool = False  # flag to check if the future is cancelled
    #is_running: bool = False  # flag to check if the future is currently running

    def __init__(self, fid, gr, src, num_vals):
        super().__init__()
        self.fid = fid
        self.gr = gr
        self.src = src
        self.nvals = num_vals
        #self.values = []
        self._result = []
        self.blocked = False
        #self.callbacks = []

    def get(self):
        """ Blocking call on current entry method's thread to obtain the values of the
            future. If the values are already available then they are returned immediately.
        """

        if not self.gotvalues:
            print("NOT AVAILABLE")
            self.blocked = True
            self.gr = getcurrent()
            self._result = threadMgr.pauseThread()
        else:
            print("AVAILABLE")

        if isinstance(self._exception, Exception):
            raise self._exception

        return self.values[0] if self.nvals == 1 else self.values

    # self.ready() == self.gotvalues == (self._state == FINISHED)?
    # Why do we need three ways to say the same thing?
    # ready() is only true if all values have been received. 
    def ready(self):
        return self.gotvalues

    @property
    def values(self):
        return self._result

    @property
    def gotvalues(self):
        return len(self.values) == self.nvals

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

        retval = False
        with self._condition:
            if self._state in {CANCELLED, CANCELLED_AND_NOTIFIED, FINISHED}:
                raise InvalidStateError('{}: {!r}'.format(self._state, self))

            self.values.append(result)

            # Should it be finished after receiving a single exception
            # or should it wait for all of them?
            if isinstance(result, Exception):
                self._exception = exception
                #self._state = FINISHED
                #for waiter in self._waiters:
                #    waiter.add_exception(self)
                #self._condition.notify_all()
                #self._invoke_callbacks()

            if self.gotvalues:
                self._state = FINISHED
                #if isinstance(self._exception, Exception)
                #    for waiter in self._waiters:
                #        waiter.add_exception(self)
                # add_exception and add_result seem to
                # do the same thing so we shouldn't
                # need to call them separately
                for waiter in self._waiters:
                    waiter.add_result(self)
                self._condition.notify_all()
                self._invoke_callbacks()
                retval = True

        #retval = False
        #self.values.append(result)
        #if isinstance(result, Exception):
        #    self._exception = result
        #if len(self.values) == self.nvals:
        #    self._state = FINISHED
            #self.gotvalues = True
            #retval = True
            #self.is_running = False
            #self._run_callbacks()

        return retval

    def resume(self, threadMgr):
        if self.blocked == 2:
            # someone is waiting for future to become ready, signal by sending
            # myself
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

    #@property
    #def _result():
    #   return self.values 

    def cancel(self):
        retval = super().cancel()
        if retval:
            threadMgr.cancelFuture(self)
        return retval
        '''
        """Cancel the future if possible.

        Returns True if the future was cancelled, False otherwise. A future
        cannot be cancelled if it is running or has already completed.
        """
        with self._condition:
            if self._state in [RUNNING, FINISHED]:
                return False

            if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:
                return True

            self._state = CANCELLED
            threadMgr.cancelFuture(self)
            self._condition.notify_all()

        self._invoke_callbacks()
        return True
        '''

    """
    def cancel(self):
        #retval = super().cancel()
        
        if self.running() or self.done():
            return False
        else:
            self.is_cancelled = True
            threadMgr.cancelFuture(self)
            self._run_callbacks()
            return True
    """
    """
    def cancelled(self):
        return self.is_cancelled

    def running(self):
        return self.is_running

    def done(self):
        print(
            "Ready:", self.ready(),
            "Cancelled:", self.cancelled(),
            "Failed:", isinstance(
                self.error,
                Exception),
            "Running:", self.running())
        return self.ready() or self.cancelled() or isinstance(self.error, Exception)
    """

    """
    def result(self, timeout=None):
        if self.cancelled():
            raise CancelledError()
        with Timeout(timeout, TimeoutError):
            result = self.get()
        return result

    def exception(self, timeout=None):
        try:
            self.result(timeout=timeout)
        except Exception as e:
            if isinstance(
                    e, TimeoutError) or isinstance(
                    e, CancelledError) or e != self.error:
                raise e

        return self.error
    """

    """
    def _run_callback(self, callback):
        try:
            callback(self)
        except Exception as e:
            print(e, file=stderr)

    def _run_callbacks(self):
        while len(self.callbacks) > 0:
            try:
                callback = self.callbacks.pop()
                self._run_callback(callback)
            except IndexError:
                break

    def add_done_callback(self, callback):
        self.callbacks.append(callback)
        # Status may have changed while appending. Run
        # any remaining callbacks
        if self.done():
            self._run_callbacks()
    """

    def set_running_or_notify_cancel(self):
        """
        if self.cancelled():
            retval = False
        elif self.done():
            raise InvalidStateError()
        else:
            self.is_running = True
            retval = True
        """

        # How to force this future to start running asynchronously?
        retval = super().set_running_or_notify_cancel()
        if retval:
            #self.waitReady(None)
            #self.resume(threadMgr)
            threadMgr.start()

        return retval

    # The following methods aren't used here, but concurrent.futures says unit
    # tests may use them

    # Add this logic (and the exception logic) to deposit
    '''
    def set_result(self, result):
        """Sets the return value of work associated with the future.
        Should only be used by Executor implementations and unit tests.
        """
        with self._condition:
            if self._state in {CANCELLED, CANCELLED_AND_NOTIFIED, FINISHED}:
                raise InvalidStateError('{}: {!r}'.format(self._state, self))
            self._result = result
            self._state = FINISHED
            for waiter in self._waiters:
                waiter.add_result(self)
            self._condition.notify_all()
        self._invoke_callbacks()
    '''

    # concurrent.futures assumes the future only needs to wait for a single
    # result, but the charm4py future can wait on multiple deposits
    def set_result(self, result):
        if isinstance(result, Iterable):
            assert len(result) == self.nvals
            for entry in result:
                self.deposit(entry)
        else:
            assert self.nvals == 1
            self.deposit(result)
        #super().set_result(exception)

    def set_exception(self, exception):
        self.deposit(exception)
        #super().set_exception(exception)

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
            raise Charm4PyError(
                'Migration of chares with active threads is not currently supported')

    def throwNotThreadedError(self):
        raise NotThreadedError(
            "Method '" +
            charm.last_em_exec.C.__name__ +
            "." +
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
        assert len(
            futures) < FIDMAXVAL, 'Too many pending futures, cannot create more'
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
            raise Charm4PyError(
                'No pending future with fid=' +
                str(fid) + '. A common reason is '
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
        #f.gotvalues = True
        f._result = [None] * f.nvals
        f.resume(self)

    # TODO: method to cancel collective future. the main issue with this is
    # that the future would need to be canceled on every chare waiting on it

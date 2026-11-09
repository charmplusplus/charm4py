#
# @author Juan Galvez (jjgalvez@illinois.edu)
#
# This package allows writing and running Charm++ applications in Python. It
# accesses the C/C++ Charm++ shared library for core runtime functionality,
# and introduces new features thanks to the dynamic language properties of Python.
#
import sys
import os
import pickle as cPickle
from io import StringIO
import inspect
import time
import gc
import ctypes
from collections import defaultdict
import traceback
from . import _ckdyn
from . import chare
from .chare import MAINCHARE, GROUP, ARRAY, CHARE, NODEGROUP, CHARM_TYPES
from .chare import Chare, Mainchare, Group, ArrayMap, Array, SingletonChare, NodeGroup
from .chare import (CKDYN_MAINCHARE, CKDYN_CHARE, CKDYN_GROUP, CKDYN_NODEGROUP,
                    CKDYN_ARRAY, CKDYN_ARRAYMAP, CKDYN_EP_THREADED, CKDYN_PUP_MIGRATE,
                    CKDYN_PUP_SIZING)
from . import entry_method
from . import threads
from .threads import Future, LocalFuture, LocalMultiFuture
from . import reduction
from . import wait
from ._ckdyn import MessageBuffer
from . import ray


def register(C):
    if ArrayMap in C.mro():
        charm.register(C, (GROUP,))  # register ArrayMap only as Group
    elif Chare in C.mro():
        charm.register(C)
    else:
        raise Charm4PyError("Class " + str(C) + " is not a Chare (can't register)")
    return C


class Options(object):

    def __str__(self):
        output = ''
        for varname in dir(self):
            var = getattr(self, varname)
            if isinstance(var, Options) or varname.startswith('__') or callable(var):
                continue
            output += varname + ': ' + str(var) + '\n'
        return output

    def check_deprecated(self):
        old_options = {'PROFILING', 'PICKLE_PROTOCOL', 'LOCAL_MSG_OPTIM',
                       'LOCAL_MSG_BUF_SIZE', 'AUTO_FLUSH_WAIT_QUEUES', 'QUIET'}
        if len(old_options.intersection(set(dir(self.__class__)))) != 0:
            raise Charm4PyError('Options API has changed. Use charm.options instead')


class Charm4PyError(Exception):
    def __init__(self, msg):
        super(Charm4PyError, self).__init__(msg)
        self.message = msg


class DevicePersistentHandle(object):
    """ The handle of a DevicePersistent buffer: immutable bytes that may be sent
        to any chare (as an argument, or inside a pickled object) and passed to
        that chare's own DevicePersistent.get or put. """
    __slots__ = ('_b',)

    def __init__(self, b):
        object.__setattr__(self, '_b', bytes(b))

    def __setattr__(self, name, value):
        raise AttributeError('DevicePersistentHandle is immutable')

    def __reduce__(self):
        return (DevicePersistentHandle, (self._b,))

    def __eq__(self, other):
        return isinstance(other, DevicePersistentHandle) and other._b == self._b

    def __hash__(self):
        return hash(self._b)

    def __repr__(self):
        return 'DevicePersistentHandle(%d bytes)' % len(self._b)


class DevicePersistent(object):
    """ A device array registered for direct get and put with other
        DevicePersistent buffers (SPEC-7-device section 2). The array must stay
        contiguous and alive until close(). get and put return a Future that
        receives None when the transfer is complete; one transfer at a time per
        buffer. Call them from a coroutine (a threaded entry method). """

    MODES = ('copy', 'ipc', 'rdma')

    def __init__(self, array):
        if not _ckdyn.has_device_support():
            raise Charm4PyError('DevicePersistent needs a CUDA or HIP build of Charm++')
        if not hasattr(array, '__cuda_array_interface__'):
            raise TypeError('DevicePersistent: expected an array with __cuda_array_interface__')
        self._p = _ckdyn.DevPersist(array)
        self.array = array
        self._handle = DevicePersistentHandle(self._p.handle())
        self._fut = None

    @property
    def handle(self):
        return self._handle

    @property
    def nbytes(self):
        return self._p.nbytes

    @property
    def closed(self):
        return self._p.closed

    def _check(self, h, what):
        if self._p.closed:
            raise Charm4PyError('DevicePersistent.' + what + ': the buffer is closed')
        if self._p.pe != charm.myPe():
            raise Charm4PyError('DevicePersistent.' + what + ': the buffer was opened on PE ' + str(self._p.pe))
        if self._fut is not None and not self._fut.ready():
            raise Charm4PyError('DevicePersistent.' + what + ': a transfer on this buffer is outstanding')
        if not isinstance(h, DevicePersistentHandle):
            raise TypeError('DevicePersistent.' + what + ': expected a DevicePersistentHandle, got ' + type(h).__name__)

    def get(self, src):
        """ copy the buffer `src` (a handle) into this one """
        self._check(src, 'get')
        f = charm.createFuture()
        self._p.get(src._b, f.fid)
        self._fut = f
        return f

    def put(self, dst):
        """ copy this buffer into the buffer `dst` (a handle) """
        self._check(dst, 'put')
        f = charm.createFuture()
        self._p.put(dst._b, f.fid)
        self._fut = f
        return f

    def mode(self, h):
        """ 'copy', 'ipc' or 'rdma': how a transfer with `h` would move the bytes """
        return DevicePersistent.MODES[self._p.mode(h._b)]

    def close(self):
        if self._p.closed:
            return
        if self._fut is not None and not self._fut.ready():
            raise Charm4PyError('DevicePersistent.close: a transfer on this buffer is outstanding')
        self._p.close()
        self.array = None

    def __del__(self):
        p = getattr(self, '_p', None)
        if p is None or p.closed:
            return
        import warnings
        if self._fut is not None and not self._fut.ready():
            warnings.warn('DevicePersistent garbage-collected with a transfer outstanding; not closed', RuntimeWarning)
            return
        warnings.warn('DevicePersistent garbage-collected without close(); closing it', RuntimeWarning)
        p.close()

# Acts as the Charm runtime at the Python level (there is one instance of this class
# per process)
class Charm(object):

    if os.name == 'nt':
        class PrintStream(object):
            def write(self, msg):
                _ckdyn.printf(msg)
            def flush(self):
                pass

    def __init__(self):
        self.started = False
        self._myPe = -1
        self._numPes = -1
        self.registered = {}      # class -> set of Charm types (Mainchare, Group, Array) for which this class is registered
        self.register_order = []  # list of classes in registration order (all processes must use same order)
        self.chares = {}
        self.groups = {}          # group ID -> group instance on this PE
        self.nodegroups = {}      # nodegroup ID -> nodegroup branch of this process (a separate id space)
        self.arrays = defaultdict(dict)  # aid -> idx -> array element instance with index idx on this PE
        self.entryMethods = {}    # ep_idx -> EntryMethod object
        self.classEntryMethods = [{} for _ in CHARM_TYPES]  # charm_type_id -> class -> list of EntryMethod objects
        self.proxyClasses      = [{} for _ in CHARM_TYPES]  # charm_type_id -> class -> proxy class
        self.sched_tagpool = set(range(1, 128))  # pool of tags for scheduling callables
        self._ckpt_future = None      # future of the charm.checkpoint(dir, resume) in progress on this PE
        self._ckpt_seq = 0            # sequence number of the last checkpoint started on this PE
        self._restart_done_seen = False  # the restart's own resume delivery arrived (restarted process)
        self.sched_next_tag = 128  # the pool grows from here when every tag is in use
        self.sched_callables = {}  # tag -> (callable, args)

        self.options = Options()
        self.options.profiling = False
        self.options.pickle_protocol = 5
        self.options.local_msg_optim = False     # accepted and ignored
        self.options.local_msg_buf_size = 50     # accepted and ignored
        self.options.copy_received_arrays = False
        self.options.zerocopy_threshold = 1 << 20
        self.options.auto_flush_wait_queues = True
        self.options.quiet = False
        self.options.remote_exec = False
        self.options.interactive = Options()
        self.options.interactive.verbose = 1
        self.options.interactive.broadcast_imports = True
        # the +libcharm_interface <name> pair is accepted and ignored (one interface exists)
        if '+libcharm_interface' in sys.argv:
            arg_idx = sys.argv.index('+libcharm_interface')
            del sys.argv[arg_idx:arg_idx + 2]
        self.reducers = reduction.ReducerContainer(self)
        self.redMgr = reduction.ReductionManager(self, self.reducers)
        self.mainchareRegistered = False
        self._mainchare_class = None
        # entry point to Charm program. can be used in place of defining a Mainchare
        self.entry_func = None
        self.last_em_exec = None
        self.interactive = False
        self.last_exception_timestamp = time.time()
        # store chare types defined after program start and other objects created
        # in interactive mode
        self.dynamic_register = sys.modules['__main__'].__dict__
        self.lb_requested = False
        self.threadMgr = threads.EntryMethodThreadManager(self)
        self.createFuture = self.Future = self.threadMgr.createFuture
        self.DevicePersistent = DevicePersistent
        self.DevicePersistentHandle = DevicePersistentHandle

        # The send buffer is not currently used since we do not buffer messages on
        # the sender. This should be used later when scheduling decisions are to be
        # based on locations of arguments
        self.send_buffer = MessageBuffer()
        self.receive_buffer = MessageBuffer()
        # TODO: maybe implement this buffer in c++
        self.future_get_buffer = {}

        #registered methods for ccs
        self.ccs_methods = {}

    def __init_profiling__(self):
        # these are attributes used only in profiling mode
        # list of Chare types that are registered and used internally by the runtime
        self.internalChareTypes = set()
        # num_msgs_sent, min_size, max_size, sum_size, last_msg_sent_size
        self.msg_send_stats = [0, int(10e6), 0, 0, -1]
        # num_msgs_rcvd, min_size, max_size, sum_size, last_msg_rcvd_size
        self.msg_recv_stats = [0, int(10e6), 0, 0, -1]
        # currently running entry method
        self.runningEntryMethod = None
        # chares created on this PE
        self.activeChares = set()
        # receive time is not measured (set by the old binding when a message arrived)
        self._precvtime = -1

    
    def print_dbg(self, *args, **kwargs):
        print("PE", self.myPe(), ":", *args, **kwargs)
    
    @entry_method.coro
    def get_future_value(self, fut):
        #self.print_dbg("Getting data for object", fut.id)
        obj = fut.lookup_object()
        if obj is None:
            local_f = LocalFuture()
            self.future_get_buffer[fut.store_id] = (local_f, fut)
            fut.request_object()
            local_f.get()
            return fut.lookup_object()
        else:
            return obj
        
    @entry_method.coro
    def getany_future_value(self, futs, num_returns):
        ready_count = 0
        ready_list = []
        not_local = []
        for f in futs:
            if f.is_local():
                ready_count += 1
                ready_list.append(f)
            else:
                f.request_object()
                not_local.append(f)
        if ready_count >= num_returns:
            return ready_list[:ready_count]
        else:
            local_f = LocalMultiFuture(num_returns - ready_count)
            for f in not_local:
                self.future_get_buffer[f.store_id] = (local_f, f)
            result = local_f.get()
            for f in not_local:
                self.future_get_buffer.pop(f.store_id, None)
            return ready_list + result
        
    def check_futures_buffer(self, obj_id):
        if obj_id in self.future_get_buffer:
            local_f, fut = self.future_get_buffer.pop(obj_id)
            local_f.send(fut)

    def check_send_buffer(self, obj_id):
        completed = self.send_buffer.check(obj_id)

    def check_receive_buffer(self, obj_id):
        #print("Received result for", obj_id, "on pe", self._myPe)
        completed = self.receive_buffer.check(obj_id)
        for args in completed:
            args = list(args)
            for i, arg in enumerate(args[-1][:-1]):
                if isinstance(arg, Future):
                    dep_obj = arg.lookup_object()
                    args[-1][i] = dep_obj
            self.invokeEntryMethod(*args, ret_fut=True)

    def handleGeneralError(self, error=None):
        """ Report an exception that escaped an entry method (the current one
            when `error` is None): print the traceback and abort, or in
            interactive mode forward it to PE 0. """
        if error is None:
            error = sys.exc_info()[1]
        if not self.interactive:
            if hasattr(error, 'remote_stacktrace'):
                origin, stacktrace = error.remote_stacktrace
                print('----------------- Python Stack Traceback PE ' + str(origin) + ' -----------------')
                print(stacktrace)
            else:
                print('----------------- Python Stack Traceback PE ' + str(self.myPe()) + ' -----------------')
                traceback.print_tb(error.__traceback__, limit=None)
            self.abort(type(error).__name__ + ': ' + str(error))
        else:
            self.thisProxy[self.myPe()].propagateException(self.prepareExceptionForSend(error))

    def prepareExceptionForSend(self, e):
        if not hasattr(e, 'remote_stacktrace'):
            f = StringIO()
            tb = e.__traceback__
            if tb is None:
                tb = sys.exc_info()[2]
            traceback.print_tb(tb, limit=None, file=f)
            e.remote_stacktrace = (self.myPe(), f.getvalue())
        return e

    def process_em_exc(self, e, obj, header):
        if b'block' not in header:
            raise e
        # remote is expecting a response via a future, send exception to the future
        blockFuture = header[b'block']
        if b'creation' in header:
            # don't send anything in this case (future is not guaranteed to be used)
            obj.contribute(None, None, blockFuture)
            raise e
        self.prepareExceptionForSend(e)
        if b'bcast' in header:
            if b'bcastret' in header:
                obj.contribute(e, self.reducers.gather, blockFuture)
            else:
                # NOTE: it will work if some elements contribute with an exception (here)
                # and some do nop (None) reduction. Charm++ will ignore the nops
                obj.contribute(e, self.reducers._bcast_exc_reducer, blockFuture)
        else:
            blockFuture.send(e)

    def recvReadOnly(self, msg):
        roData = cPickle.loads(msg)
        for name, obj in roData.items():
            if name == 'charm_pool_proxy__h':
                from .pool import Pool
                self.pool = Pool(obj)
            else:
                setattr(readonlies, name, obj)
        gc.collect()

    def invokeEntryMethod(self, obj, ep, header, args, ret_fut=False):
        em = self.entryMethods[ep]
        if (em.when_cond is not None) and (not em.when_cond.evaluateWhen(obj, args)):
            obj.__waitEnqueue__(em.when_cond, (0, em, header, args))
        else:
            em.run(obj, header, args, ret_fut=ret_fut)
            if self.options.auto_flush_wait_queues and obj._cond_next is not None:
                obj.__flush_wait_queues__()

    # ---------------- targets of the _ckdyn hooks (SPEC-4a 2.2) ----------------

    def _mainchare(self, target, argv):
        """ create the main chare on PE 0; returns it """
        cid = target.cid
        assert cid not in self.chares, 'Chare ' + str(cid) + ' already instantiated'
        em = self.classEntryMethods[MAINCHARE][self._mainchare_class][0]
        assert em.name == '__init__', 'Specified mainchare entry method is not constructor'
        self._createInternalChares()
        obj = object.__new__(em.C)  # create object but don't call __init__
        obj._ckdyn_target = target
        Mainchare.initMember(obj, target)
        Chare.__init__(obj)  # Chare state first; super(em.C, obj) would be a parent user class
        if self.entry_func is not None:
            assert isinstance(obj, chare.DefaultMainchare)
            obj.main = self.entry_func
        if self.options.profiling:
            self.activeChares.add((em.C, Mainchare))
        gc.collect()
        try:
            em.run(obj, {}, [argv])  # now call the user's __init__
        except Exception as e:
            self.handleGeneralError(e)
        self.chares[cid] = obj
        gc.collect()
        return obj

    def _pack_readonlies(self):
        """ the readonly attributes, pickled (on PE 0, after the main chare's constructor) """
        roData = {}
        for attr in dir(readonlies):  # attr is string
            if not attr.startswith('_') and not attr.endswith('_'):
                roData[attr] = getattr(readonlies, attr)
        return cPickle.dumps(roData, self.options.pickle_protocol)

    def _construct(self, target, ep, header, args):
        """ create a group branch, array element or singleton chare; returns it """
        em = self.entryMethods[ep]
        kind = target.kind
        if b'block' in header and b'bcast' in header:
            # group or array created from a coroutine: the constructor's slow path
            # contributes a nop reduction to the creation future when __init__ has
            # finished (EntryMethod._run); a failing constructor contributes None
            # (process_em_exc). CKDYN_MSG_CREATION is not used.
            header[b'creation'] = True
        obj = object.__new__(em.C)  # create object but don't call __init__
        obj._ckdyn_target = target
        if kind == CKDYN_NODEGROUP:
            if self.options.profiling:
                self.activeChares.add((em.C, NodeGroup))
            NodeGroup.initMember(obj, target.gid, _ckdyn.my_node())
            Chare.__init__(obj)
            self.nodegroups[target.gid] = obj
        elif kind in (CKDYN_GROUP, CKDYN_ARRAYMAP):
            if self.options.profiling:
                self.activeChares.add((em.C, Group))
            Group.initMember(obj, target.gid)
            Chare.__init__(obj)  # Chare state first; super(em.C, obj) would be a parent user class
            if b'constrained' in header:
                # Group(C, onPEs=...): thisProxy becomes the group section proxy
                # when this branch joins it (_section_joined)
                obj._constrained = True
            self.groups[target.gid] = obj
        elif kind == CKDYN_ARRAY:
            if self.options.profiling:
                self.activeChares.add((em.C, Array))
            index = tuple(target.index)
            Array.initMember(obj, target.aid, index, single=(b'single' in header))
            Chare.__init__(obj)
            self.arrays[target.aid][index] = obj
        elif kind == CKDYN_CHARE:
            if self.options.profiling:
                self.activeChares.add((em.C, SingletonChare))
            SingletonChare.initMember(obj, target)
            Chare.__init__(obj)
            self.chares[target.cid] = obj
        else:
            raise Charm4PyError('Cannot construct a chare of CkDyn kind ' + str(kind))
        try:
            em.run(obj, header, args)  # now call the user's __init__
        except Exception as e:
            self.handleGeneralError(e)
        return obj

    def _invoke(self, obj, ep, header, args):
        """ slow delivery path: when conditions, coroutines, header flags, profiling """
        try:
            if b'refnum' in header:
                if self.entryMethods[ep].name != '_coll_future_deposit_result':
                    # the done callback of charm.checkpoint(dir, resume): the
                    # reference number is its tag, not an argument of the entry
                    self._checkpoint_done(args.pop(0))
                elif b'sid' in header:
                    # the result of a section allreduce: its collective future is
                    # keyed by (reduction number, section id) (Chare.allreduce)
                    args[0] = (args[0], header[b'sid'])
            if b'custom_reducer' in header:
                # reduction result of a Python reducer: the result is the last argument
                reducer = self.redMgr.reducers_by_id[header[b'custom_reducer']]
                if reducer.hasPostprocess:
                    args[-1] = reducer.postprocess(args[-1])
            if header.get('is_ray', False):
                dep_ids = []
                for i, arg in enumerate(args[:-1]):
                    if isinstance(arg, Future):
                        dep_obj = arg.lookup_object()
                        if dep_obj is None:
                            dep_ids.append(arg.store_id)
                            arg.request_object()
                        else:
                            args[i] = dep_obj
                if len(dep_ids) > 0:
                    self.receive_buffer.insert(dep_ids, (obj, ep, header, args))
                else:
                    self.invokeEntryMethod(obj, ep, header, args, ret_fut=True)
            else:
                self.invokeEntryMethod(obj, ep, header, args)
        except Exception as e:
            self.handleGeneralError(e)

    def _process_exception(self, obj, ep, exc):
        """ an exception raised by an entry method on the fast path """
        if isinstance(exc, SystemExit):
            self.exit(entry_method.exit_code(exc))
        else:
            self.handleGeneralError(exc)

    def _deposit_future(self, fid, value, rid):
        try:
            if rid >= 0:
                reducer = self.redMgr.reducers_by_id[rid]
                if reducer.hasPostprocess:
                    value = reducer.postprocess(value)
            self.threadMgr.depositFuture(fid, value)
        except Exception as e:
            self.handleGeneralError(e)

    def _reduce_foreign(self, rid, values):
        return self.redMgr.reducers_by_id[rid](values)

    def _pack(self, obj, reason):
        """ pickled obj for migration or checkpoint; None refuses (a coroutine of obj is
            suspended). reason may include CKDYN_PUP_SIZING: the bytes of a sizing pass
            are kept on obj and returned by the packing pass that follows. A sizing
            pass can also run alone (AtSync under +MetaLB sizes every element), so it
            never refuses and changes nothing else; the element leaves charm.arrays in
            _destruct, when the runtime deletes it after it has migrated. """
        sizing = reason & CKDYN_PUP_SIZING
        why = reason & 0xff
        d = obj.__dict__
        cached = d.pop('_ckdyn_packed', None)
        if why == CKDYN_PUP_MIGRATE and obj._nsuspended > 0:
            if not sizing:
                return None
            # sizing alone, or the sizing pass of a migration that the packing pass
            # refuses: report the size only
        elif not sizing and cached is not None and cached[0] == why:
            return cached[1]
        if hasattr(obj, '__channels__'):
            assert len(obj.__pendingChannels__) == 0, 'Cannot migrate chares that did not complete channel establishment'
        target = d.pop('_ckdyn_target', None)
        try:
            data = cPickle.dumps(obj, self.options.pickle_protocol)
        except Exception as e:
            raise Charm4PyError('cannot pickle ' + type(obj).__name__ + str(getattr(obj, 'thisIndex', '')) +
                                (' for migration' if why == CKDYN_PUP_MIGRATE else ' for a checkpoint') +
                                ': ' + type(e).__name__ + ': ' + str(e)) from e
        finally:
            if target is not None:
                obj._ckdyn_target = target
        if sizing and not (why == CKDYN_PUP_MIGRATE and obj._nsuspended > 0):
            d['_ckdyn_packed'] = (why, data)
        return data

    def _unpack(self, target, reason, data):
        """ an object arriving by migration or restored from a checkpoint; returns it """
        obj = cPickle.loads(data)
        obj._ckdyn_target = target
        if reason != CKDYN_PUP_MIGRATE:
            # restored from a checkpoint: its coroutines were not restored
            obj._numthreads = 0
            obj._nsuspended = 0
            if isinstance(obj, CharmRemote):
                self.thisProxy = obj.thisProxy
        kind = target.kind
        if kind == CKDYN_ARRAY:
            self.arrays[target.aid][tuple(target.index)] = obj
        elif kind == CKDYN_NODEGROUP:
            self.nodegroups[target.gid] = obj
        elif kind in (CKDYN_GROUP, CKDYN_ARRAYMAP):
            self.groups[target.gid] = obj
        else:
            self.chares[target.cid] = obj
        if reason == CKDYN_PUP_MIGRATE:
            ems = self.classEntryMethods[ARRAY].get(obj.__class__, ())
            try:
                for em in ems:
                    if em.name == 'migrated':
                        em.run(obj, {}, ())
                        break
                else:
                    obj.migrated()
            except Exception as e:
                self.handleGeneralError(e)
        return obj

    def _map_proc_num(self, obj, aid, index):
        return obj.procNum(index)

    def _resume_from_sync(self, obj):
        # not called inline: with no load balancer AtSync calls ResumeFromSync
        # synchronously, and a resumeFromSync that calls AtSync would recurse
        if hasattr(obj, 'resumeFromSync'):
            try:
                obj.thisProxy[obj.thisIndex].resumeFromSync()
            except Exception as e:
                self.handleGeneralError(e)

    def _is_local(self, obj):
        """ obj is the live local object of its target (not migrated away or destroyed) """
        target = getattr(obj, '_ckdyn_target', None)
        if target is None:
            return False
        kind = target.kind
        if kind == CKDYN_ARRAY:
            return self.arrays.get(target.aid, {}).get(tuple(target.index)) is obj
        if kind in (CKDYN_GROUP, CKDYN_ARRAYMAP, CKDYN_NODEGROUP):
            return self.groups.get(target.gid) is obj
        return self.chares.get(target.cid) is obj

    def _entry_of(self, obj, name):
        """ (entry index, argument count) of method `name` of obj's class in obj's role """
        kind = obj._ckdyn_target.kind
        tid = {CKDYN_ARRAY: ARRAY, CKDYN_MAINCHARE: MAINCHARE, CKDYN_CHARE: CHARE}.get(kind, GROUP)
        for em in self.classEntryMethods[tid].get(obj.__class__, ()):
            if em.name == name:
                return em.epIdx, chare.getEntryMethodInfo(em.C, name)[0]
        raise Charm4PyError(str(name) + ' is not an entry method of ' + obj.__class__.__name__)

    def _destruct(self, obj):
        """ the runtime deleted obj (migrated away or destroyed) """
        target = getattr(obj, '_ckdyn_target', None)
        if target is not None:
            kind = target.kind
            if kind == CKDYN_ARRAY:
                elems = self.arrays.get(target.aid)
                index = tuple(target.index)
                if elems is not None and elems.get(index) is obj:
                    del elems[index]
            elif kind == CKDYN_NODEGROUP:
                if self.nodegroups.get(target.gid) is obj:
                    del self.nodegroups[target.gid]
            elif kind in (CKDYN_GROUP, CKDYN_ARRAYMAP):
                if self.groups.get(target.gid) is obj:
                    del self.groups[target.gid]
            elif self.chares.get(target.cid) is obj:
                del self.chares[target.cid]
        # facilitate garbage collection (remove cyclical references)
        obj.__dict__.pop('_active_grp_conds', None)
        obj._cond_next = None
        obj._cond_last = None

    def _error(self, msg):
        print('CkDyn error: ' + str(msg), file=sys.stderr)

    def _section_joined(self, obj, sid):
        """ obj received the creation multicast of section sid (hook section_joined,
            before obj's contribution that completes the section's creation) """
        secs = obj.__dict__.get('_sections')
        if secs is None:
            secs = obj._sections = set()
        secs.add(sid)
        if obj.__dict__.pop('_constrained', False):
            # a branch of a constrained group: the first section it joins is the
            # group section that Group(C, onPEs=...) creates right after the group
            obj.thisProxy = obj.thisProxy._as_section(sid)

    # register class C in Charm
    def registerInCharmAs(self, C, charm_type):
        charm_type_id = charm_type.type_id
        entryMethods = self.classEntryMethods[charm_type_id][C]
        if charm_type_id == MAINCHARE:
            kind = CKDYN_MAINCHARE
        elif charm_type_id == CHARE:
            kind = CKDYN_CHARE
        elif charm_type_id == GROUP:
            kind = CKDYN_ARRAYMAP if ArrayMap in C.mro() else CKDYN_GROUP
        elif charm_type_id == NODEGROUP:
            kind = CKDYN_NODEGROUP
        else:
            kind = CKDYN_ARRAY  # ndims 0: any dimension, fixed by each array's creation
        cname = chare.trace_class_name(C)
        entries = []
        for em in entryMethods:
            if charm_type_id == ARRAY and em.name in ('migrated', 'AtSync'):
                func = None  # placeholders, never sent
            else:
                func = getattr(C, em.name)
            flags = 0
            if hasattr(func, '_ck_coro'):
                flags |= CKDYN_EP_THREADED
            if func is not None:
                flags |= entry_method.entry_flags(C, em.name, kind == CKDYN_NODEGROUP)
            if func is not None and hasattr(func, '__code__'):
                argcount, argnames, defaults = chare.getEntryMethodInfo(C, em.name)
            else:
                argcount, argnames, defaults = 0, (), ()
            entries.append((func, em, cname + '.' + em.name, flags,
                            em.when_cond is not None, argcount, argnames, defaults))
        C.idx[charm_type_id], startEpIdx = _ckdyn.register_class(
            C, kind, 0, entries, cname + chare.TRACE_KIND_SUFFIX[charm_type_id])
        for i, em in enumerate(entryMethods):
            em.epIdx = startEpIdx + i
            self.entryMethods[em.epIdx] = em
        proxyClass = charm_type.__getProxyClass__(C)
        # save proxy class in the same module as its Chare class
        proxyClass.__module__ = C.__module__
        setattr(sys.modules[C.__module__], proxyClass.__name__, proxyClass)
        self.proxyClasses[charm_type_id][C] = proxyClass
        _ckdyn.set_proxy_class(C.idx[charm_type_id], proxyClass)

    def registerInCharm(self, C):
        C.idx = [None] * len(CHARM_TYPES)
        charm_types = self.registered[C]
        if Mainchare in charm_types:
            self.registerInCharmAs(C, Mainchare)
        if Group in charm_types:
            self.registerInCharmAs(C, Group)
        if Array in charm_types:
            self.registerInCharmAs(C, Array)
        if SingletonChare in charm_types:
            self.registerInCharmAs(C, SingletonChare)
        if NodeGroup in charm_types:
            self.registerInCharmAs(C, NodeGroup)

    # first callback from the Charm++ runtime (hook register_classes)
    # this method registers classes with CkDyn
    def registerMainModule(self):
        self._myPe   = _ckdyn.my_pe()
        self._numPes = _ckdyn.num_pes()

        # Charm++ library captures stdout/stderr. here we reset the streams with a buffering
        # policy that ensures that messages reach Charm++ in a timely fashion
        if os.name == 'nt':
            sys.stdout = Charm.PrintStream()
        else:
            sys.stdout = os.fdopen(1, 'wt', 1)
            sys.stderr = os.fdopen(2, 'wt', 1)

        if (self.myPe() == 0) and (not self.options.quiet):
            import platform
            from . import charm4py_version
            py_impl = platform.python_implementation()
            print("Charm4py> Running Charm4py version " + charm4py_version +
                       " on Python " + str(platform.python_version()) + " (" +
                       py_impl + "). Using 'ckdyn' interface to access Charm++")
            if py_impl != 'CPython':
                raise Charm4PyError('PyPy is no longer supported. Use CPython instead')
            if sys.version_info < (3,8,0):
                raise Charm4PyError('Python 2 is no longer supported. Use Python 3.8 or above instead')
            if self.options.profiling:
                print('Charm4py> Profiling is ON (this affects performance)')
            if self.options.local_msg_optim:
                print('Charm4py> charm.options.local_msg_optim is ignored (local messages always go through Charm++)')

        for C in self.register_order:
            self.registerInCharm(C)

    def registerAs(self, C, charm_type_id):
        if charm_type_id == MAINCHARE:
            assert not self.mainchareRegistered, 'More than one entry point has been specified'
            self.mainchareRegistered = True
            self._mainchare_class = C
            # make mainchare constructor always a coroutine
            if sys.version_info < (3, 0, 0):
                entry_method.coro(C.__init__.im_func)
            else:
                entry_method.coro(C.__init__)
        charm_type = chare.charm_type_id_to_class[charm_type_id]
        # print("charm4py: Registering class " + C.__name__, "as", charm_type.__name__, "type_id=", charm_type_id, charm_type)
        profilingOn = self.options.profiling
        ems = [entry_method.EntryMethod(C, m, profilingOn) for m in charm_type.__baseEntryMethods__()]
        ems += [entry_method.EntryMethod(C, m, profilingOn) for m in self._entry_method_names(C, charm_type_id)]
        self.classEntryMethods[charm_type_id][C] = ems
        self.registered[C].add(charm_type)

    def _entry_method_names(self, C, charm_type_id):
        """ the user entry methods of C as charm_type_id, besides __baseEntryMethods__ """
        names = []
        for m in dir(C):
            m_obj = getattr(C, m)
            if not callable(m_obj) or inspect.isclass(m_obj):
                continue
            if m in chare.method_restrictions['reserved'] and m_obj != getattr(Chare, m):
                raise Charm4PyError(str(C) + " redefines reserved method '"  + m + "'")
            if m.startswith('__') and m.endswith('__'):
                continue  # filter out non-user methods
            if m in chare.method_restrictions['non_entry_method']:
                continue
            if charm_type_id != ARRAY and m in {'migrate', 'setMigratable', 'ckDestroy'}:
                continue
            names.append(m)
        return names

    def _check_entry_attrs(self, C, collections):
        """ raise Charm4PyError for an @entry attribute that registration would refuse """
        for charm_type_id in collections:
            for m in ['__init__'] + self._entry_method_names(C, charm_type_id):
                entry_method.entry_flags(C, m, charm_type_id == NODEGROUP)

    # called by user (from Python) to register their Charm++ classes with the Charm4py runtime
    # by default a class is registered to work as Group, Array, singleton Chare and NodeGroup;
    # a class with a nodegroup-only entry attribute (@entry(exclusive=True) or immediate)
    # only as NodeGroup
    def register(self, C, collections=None):
        if C in self.registered:
            return
        if (not hasattr(C, 'mro')) or (Chare not in C.mro()):
            raise Charm4PyError('Only subclasses of Chare can be registered')
        if collections is None:
            if entry_method.has_nodegroup_only_attrs(C):
                collections = (NODEGROUP,)
            else:
                collections = (GROUP, ARRAY, CHARE, NODEGROUP)

        # cache of template condition objects for `chare.wait(cond_str)` calls
        # maps cond_str to condition object. the condition object stores the lambda function associated with cond_str
        # TODO: remove old/unused condition strings from dict?
        C.__charm_wait_conds__ = {}
        self._check_entry_attrs(C, collections)
        self.registered[C] = set()
        for charm_type_id in collections:
            self.registerAs(C, charm_type_id)
        self.register_order.append(C)

    def _registerInternalChares(self):
        self.register(CharmRemote, (GROUP,))

        from .pool import PoolScheduler, Worker
        if self.interactive:
            entry_method.coro(PoolScheduler.start)
            entry_method.coro(PoolScheduler.startSingleTask)
        self.register(PoolScheduler, (ARRAY,))
        self.register(Worker, (GROUP,))

        if self.options.profiling:
            self.internalChareTypes.update({CharmRemote, PoolScheduler, Worker})

    def _createInternalChares(self):
        Group(CharmRemote)

        from .pool import Pool, PoolScheduler
        # one-element array form: pool.py needs an element proxy (elemIdx a tuple)
        pool_proxy = Chare(PoolScheduler, onPE=0, migratable=True)
        self.pool = Pool(pool_proxy)
        readonlies.charm_pool_proxy__h = pool_proxy

    def start(self, entry=None, classes=[], modules=[], interactive=False):
        """
        Start Charm4py program.

        IMPORTANT: classes must be registered in the same order on all processes. In
        other words, the arguments to this method must have the same ordering on all
        processes.

        Args:
            entry:   program entry point (function or Chare class)
            classes: list of Charm classes to register with runtime
            modules: list of names of modules containing Charm classes (all of the Charm
                     classes defined in the module will be registered). start will
                     always search module '__main__' for Charm classes even if no
                     arguments are passed to this method.
        """

        # TODO: remove in a future release
        self.options.check_deprecated()

        if interactive:
            from .interactive import InteractiveConsole as entry
            from .channel import Channel
            self.options.remote_exec = True
            self.origStdinFd = os.dup(0)
            self.origStoutFd = os.dup(1)
            self.interactive = True
            self.dynamic_register.update({'charm': charm, 'Chare': Chare, 'Group': Group,
                                          'Array': Array, 'Reducer': self.reducers,
                                          'threaded': entry_method.coro, 'coro': entry_method.coro,
                                          'Channel': Channel})

        if self.started:
            raise Charm4PyError('charm.start() can only be called once')
        self.started = True

        if self.options.profiling:
            self.__init_profiling__()
            self.contribute = profile_send_function(self.contribute)
            self.triggerCallableEM = entry_method.EntryMethod(self.__class__,
                                                              'triggerCallable',
                                                              True)
        if self.options.quiet and '++quiet' not in sys.argv:
            sys.argv += ['++quiet']
        elif '++quiet' in sys.argv:
            self.options.quiet = True

        self._registerInternalChares()

        if hasattr(entry, 'mro') and Chare in entry.mro():
            if entry.__init__.__code__.co_argcount != 2:
                raise Charm4PyError('Mainchare constructor must take one (and only one) parameter')
            self.register(entry, (MAINCHARE,))
        else:
            assert callable(entry), 'Given entry point is not a function or Chare'
            if entry.__code__.co_argcount != 1:
                raise Charm4PyError('Main function must have one (and only one) parameter')
            self.entry_func = entry
            self.register(chare.DefaultMainchare, (MAINCHARE,))

        for C in classes:
            if ArrayMap in C.mro():
                self.register(C, (GROUP,))  # register ArrayMap only as Group
            elif Chare in C.mro():
                self.register(C)
            else:
                raise Charm4PyError("Class", C, "is not a Chare (can't register)")

        import importlib
        M = list(modules)
        if '__main__' not in M:
            M.append('__main__')
        for module_name in M:
            if module_name not in sys.modules:
                importlib.import_module(module_name)
            for C_name, C in inspect.getmembers(sys.modules[module_name], inspect.isclass):
                if C.__module__ != chare.__name__ and hasattr(C, 'mro'):
                    if ArrayMap in C.mro():
                        self.register(C, (GROUP,))  # register ArrayMap only as Group
                    elif Chare in C.mro():
                        self.register(C)
                    elif Group in C.mro() or Array in C.mro() or Mainchare in C.mro():
                        raise Charm4PyError('Chares must not inherit from Group, Array or'
                                            ' Mainchare. Refer to new API')

        for module in (chare, entry_method, wait):
            module.charmStarting()
        self.threadMgr.start()

        self.lb_requested = '+balancer' in sys.argv
        _ckdyn.init(self, self.options)
        _ckdyn.start(sys.argv)

    # Charm class contribute function used by Array, Group and sections for reductions.
    # 'section' is a section proxy or a section id (proxy.section[1]); a branch of a
    # constrained group contributes to its group section when section is None.
    def contribute(self, data, reducer, target, chare, section=None):
        if section is None and chare.thisProxy.issec:
            section = chare.thisProxy
        rtype, data = self.redMgr.prepare(data, reducer, chare)
        if section is None:
            _ckdyn.contribute(chare, data, rtype, target)
        else:
            sid = self._section_id(section)
            if sid not in chare.__dict__.get('_sections', ()):
                raise Charm4PyError('Section reduction by ' + chare.__class__.__name__ + str(chare.thisIndex) +
                                    ', which is not a member of the section')
            _ckdyn.contribute_section(chare, sid, data, rtype, target)

    # ---------------- sections (SPEC-4b.md) ----------------

    def _section_id(self, section):
        """ the 16-byte section id of a section proxy, or section itself if it is one """
        if isinstance(section, bytes):
            return section
        if getattr(section, 'issec', False):
            return section.section[1]
        raise TypeError('expected a section proxy or a section id, got ' + repr(section))

    def _require_coroutine(self, what):
        if self.threadMgr.isMainThread():
            raise Charm4PyError(what + ' must be called from a coroutine: it waits until '
                                'every member has joined the new section(s)')

    def _create_section(self, proxy, members, done):
        """ Create one section on this PE (its root) from members, a list of
            (gid, pe) or (aid, index); returns the section id. Members from several
            arrays (the split of a combined section) become one section per array,
            combined. """
        if proxy._kind == CKDYN_GROUP:
            gid = members[0][0]
            return _ckdyn.section_create_group(gid, sorted({pe for _, pe in members}), 0, done)
        by_aid = {}
        for aid, index in members:
            by_aid.setdefault(aid, []).append(index)
        parts = []
        for aid, indices in by_aid.items():
            ndims = len(indices[0])
            flat = [i for index in indices for i in index]
            parts.append((aid, ndims, flat))
        if len(parts) == 1:
            aid, ndims, flat = parts[0]
            return _ckdyn.section_create_array(aid, ndims, flat, 0, done)
        sids = [_ckdyn.section_create_array(aid, ndims, flat, 0, None) for aid, ndims, flat in parts]
        return _ckdyn.section_combine(sids, done)

    def split(self, proxy, numsections, section_func=None, elems=None, slicing=None):
        """ Split a group, an array or a section into numsections sections; returns the
            list of section proxies. Membership is given by section_func(obj) (a section
            number, a list of them, or a negative number for none), by elems (one list
            of indices or PEs per section), by slicing (one slice per dimension; one
            section), or, with none of them, every member is in the one section.
            Must be called from a coroutine: it returns when every member of every new
            section has joined it, so sends to the returned proxies are safe at once. """
        if (not isinstance(proxy, _ckdyn.Proxy) or proxy._kind not in (CKDYN_GROUP, CKDYN_ARRAY)
                or proxy._has_elem):
            raise Charm4PyError('charm.split: expected a group, array or section proxy (not an element proxy)')
        self._require_coroutine('charm.split')
        if elems is not None:
            if numsections == 1 and not isinstance(elems[0], list) and not isinstance(elems[0], set):
                elems = [elems]
            if len(elems) != numsections:
                raise Charm4PyError('charm.split: elems has ' + str(len(elems)) + ' lists for ' +
                                    str(numsections) + ' sections')
        members = [[] for _ in range(numsections)]
        if proxy._kind == CKDYN_GROUP and not proxy.issec and elems is not None:
            # the members of a group section are PEs: no vote needed
            for n, pes in enumerate(elems):
                members[n] = [(proxy.gid, int(pe)) for pe in pes]
        else:
            # every member of the collection or section says which sections it is in
            f = self.Future()
            proxy._sectionVote_(numsections, section_func, slicing, elems, f,
                                proxy if proxy.issec else None)
            for coll_id, index, secs in f.get():
                for n in secs:
                    members[n].append((coll_id, index))
        for n, m in enumerate(members):
            if len(m) == 0:
                raise Charm4PyError('charm.split: section ' + str(n) + ' has no members')
        done = self.Future(numsections)
        sids = [self._create_section(proxy, m, done) for m in members]
        done.get()
        return [proxy._as_section(sid) for sid in sids]

    def combine(self, *proxies):
        """ The union of array sections and arrays of one chare class with one index
            dimension count, as a new section (each element once); returns its proxy.
            Group sections cannot be combined (TypeError): a combined section is a
            Charm++ multi-array section. Must be called from a coroutine. """
        if len(proxies) == 0:
            raise Charm4PyError('charm.combine: no proxies')
        first = proxies[0]
        for p in proxies:
            if not isinstance(p, _ckdyn.Proxy) or p._kind != CKDYN_ARRAY or p._has_elem:
                raise TypeError('charm.combine accepts array sections and arrays only, got ' + repr(p) +
                                ' (group sections cannot be combined)')
            if type(p) is not type(first):
                raise TypeError('charm.combine: the arrays must be of one chare class (' +
                                type(first).__name__ + ', ' + type(p).__name__ + ')')
            if p.ndims != first.ndims:
                raise TypeError('charm.combine: the arrays must have one index dimension count (' +
                                str(first.ndims) + ', ' + str(p.ndims) + ')')
        self._require_coroutine('charm.combine')
        sids = []
        for p in proxies:
            if not p.issec:
                p = self.split(p, 1)[0]  # the whole array as a section
            sids.append(p.section[1])
        done = self.Future()
        sid = _ckdyn.section_combine(sids, done)
        done.get()
        return first._as_section(sid)

    def startQD(self, callback):
        _ckdyn.start_qd(callback)

    def checkpoint(self, dirname, resume=None):
        """ Write a checkpoint of every chare to directory dirname (Charm++
            CkStartCheckpoint). A program started with +restart dirname
            continues from it.

            resume is None: returns a future that receives None once the
            checkpoint is written; call this from a coroutine and wait on the
            future. A restarted run has nothing to continue in this form.

            resume is a bound entry method of a group proxy, an array element
            proxy or the main chare (for example self.thisProxy.resumed in the
            main chare): the runtime calls that entry with no arguments once
            the checkpoint is written, and, in a program restarted from
            dirname, on the restored object once the restart completes. This
            is how a program continues after a restart, because coroutines
            (including a coroutine main) are not restored. Called from a
            coroutine, it also returns a future that receives None on the
            original run once the checkpoint is written; called from a plain
            entry method it returns None.

            Futures and suspended coroutines from before a restart do not
            exist in the restarted run.
        """
        if resume is None:
            f = self.threadMgr.createFuture()  # raises NotThreadedError outside a coroutine
            _ckdyn.checkpoint(dirname, f)
            return f
        # the done callback is resume with reference number (seq << 24) | (PE + 1);
        # charm._invoke strips it and sends _checkpoint_written to this PE
        self._ckpt_seq = (self._ckpt_seq % 255) + 1
        tag = (self._ckpt_seq << 24) | (self._myPe + 1)
        cb = _ckdyn.Callback.target(resume, tag)
        f = None
        if not self.threadMgr.isMainThread():
            f = self.threadMgr.createFuture()
        self._ckpt_future = f
        _ckdyn.checkpoint(dirname, cb)
        return f

    def _checkpoint_done(self, tag):
        """ a charm.checkpoint resume entry is being delivered with its tag """
        if _ckdyn.is_restarting() and not self._restart_done_seen:
            # the restart's own delivery: the checkpointing PE's future is gone
            self._restart_done_seen = True
            return
        pe = (tag & 0xffffff) - 1
        if pe == self._myPe:
            self._checkpoint_written(tag)
        else:
            self.thisProxy[pe]._checkpoint_written(tag)

    def _checkpoint_written(self, tag):
        """ on the PE that called charm.checkpoint(dir, resume): deposit its future """
        f = self._ckpt_future
        if f is None or (tag >> 24) != self._ckpt_seq:
            return  # a group resume delivers once per PE: only the first one counts
        self._ckpt_future = None
        self.threadMgr.depositFuture(f.fid, None)

    def waitQD(self):
        f = self.Future()
        f.ignorehang = True
        self.startQD(f)
        f.get()

    def sleep(self, secs):
        if self.threadMgr.isMainThread():
            time.sleep(secs)
        else:
            f = self.Future()
            f.ignorehang = True
            self.scheduleCallableAfter(f, secs)
            f.get()

    def awaitCreation(self, *proxies):
        for proxy in proxies:
            if not hasattr(proxy, 'creation_future'):
                if not proxy.__class__.__name__.endswith('Proxy'):
                    raise Charm4PyError('Did not pass a proxy to awaitCreation? ' + str(type(proxy)))
                raise Charm4PyError('awaitCreation can only be used if creation triggered from a coroutine entry method')
            proxy.creation_future.get()
            del proxy.creation_future

    def scheduleCallableAfter(self, callable_obj, secs, args=[]):
        if self.sched_tagpool:
            tag = self.sched_tagpool.pop()
        else:
            # every tag is held by a callable that has not run yet (a long
            # timeout holds its tag until it fires): add one instead of failing
            tag = self.sched_next_tag
            self.sched_next_tag += 1
        self.sched_callables[tag] = (callable_obj, args)
        _ckdyn.call_after(tag, secs * 1000)

    def triggerCallable(self, tag):
        try:
            if self.options.profiling:
                self.triggerCallableEM.startMeasuringTime()
            cb, args = self.sched_callables.pop(tag)
            self.sched_tagpool.add(tag)
            cb(*args)
            if self.options.profiling:
                self.triggerCallableEM.stopMeasuringTime()
        except Exception as e:
            self.handleGeneralError(e)

    # generator that yields objects (works for Futures and Channels) as they
    # become ready (have a msg ready to receive immediately)
    def iwait(self, objs):
        n = len(objs)
        f = LocalFuture()
        for obj in objs:
            if obj.ready():
                n -= 1
                yield obj
            else:
                obj.waitReady(f)
        while n > 0:
            obj = self.threadMgr.pauseThread()
            n -= 1
            yield obj

    def wait(self, objs):
        for o in self.iwait(objs):
            pass

    def recordSend(self, size):
        self.recordSendRecv(self.msg_send_stats, size)

    def recordReceive(self, size):
        self.recordSendRecv(self.msg_recv_stats, size)

    def recordSendRecv(self, stats, size):
        stats[0] += 1
        stats[1] = min(size, stats[1])
        stats[2] = max(size, stats[2])
        stats[3] += size
        stats[4] = size
        
    def __printTable__(self, table, sep):
        col_width = [max(len(x) for x in col) for col in zip(*table)]
        for j, line in enumerate(table):
            if j in sep: print(sep[j])
            print("| " + " | ".join("{:{}}".format(x, col_width[i]) for i, x in enumerate(line)) + " |")

    def printStats(self):
        """ Print the profiling tables of this PE (charm.options.profiling).
            Columns: time inside entry methods ('em', which includes the time
            spent sending, since sends made through proxies are not timed
            separately by the CkDyn binding), time recorded as receive
            overhead ('recv') and their total. """
        assert self.started, 'charm was not started'
        if not self.options.profiling:
            print('NOTE: called charm.printStats() but profiling is disabled')
            return

        em = self.runningEntryMethod
        if em is not None:
            # record elapsed time of current entry method so that it is part of displayed stats
            assert not em.measuringSendTime
            em.stopMeasuringTime()
            em.startMeasuringTime()

        def em_vals(times):
            return [times[0] + times[1], times[2], sum(times)]

        print('Timings for PE', self.myPe(), ':')
        table = [['', 'em', 'recv', 'total']]
        lineNb = 1
        sep = {}
        row_totals = [0.0] * 3
        chares_sorted = sorted([(C.__module__, C.__name__,
                                 charm_type.type_id, C, charm_type)
                                 for C, charm_type in self.activeChares])
        for _, _, _, C, charm_type in chares_sorted:
            if C in self.internalChareTypes:
                totaltime = 0.0
                for em in self.classEntryMethods[charm_type.type_id][C]:
                    if em.name == '__init__':
                        continue
                    totaltime += sum(em.times)
                if totaltime < 0.001:
                    continue
            sep[lineNb] = '------ ' + str(C) + ' as ' + charm_type.__name__ + ' ------'
            for em in self.classEntryMethods[charm_type.type_id][C]:
                if not hasattr(em, 'times'):
                    continue
                if C == chare.DefaultMainchare and self.entry_func is not None and em.name == '__init__':
                    em_name = self.entry_func.__module__ + '.' + self.entry_func.__name__ + ' (main function)'
                else:
                    em_name = em.name
                vals = em_vals(em.times)
                for i in range(len(row_totals)):
                    row_totals[i] += vals[i]
                table.append([em_name] + [str(round(v, 3)) for v in vals])
                lineNb += 1
        sep[lineNb] = '-----------------------------------------------------------'
        table.append([''] + [str(round(v, 3)) for v in row_totals])
        lineNb += 1
        triggerCallableTotalTime = sum(self.triggerCallableEM.times)
        if triggerCallableTotalTime > 0:
            sep[lineNb] = '-----------------------------------------------------------'
            vals = em_vals(self.triggerCallableEM.times)
            for i, v in enumerate(vals):
                row_totals[i] += v
            table.append(['triggerCallable'] + [str(round(v, 3)) for v in vals])
            lineNb += 1
            sep[lineNb] = '-----------------------------------------------------------'
            table.append([''] + [str(round(v, 3)) for v in row_totals])
            lineNb += 1
        self.__printTable__(table, sep)

        for i in (0, 1):
            if i == 0:
                stats, what = self.msg_send_stats, 'sent'
            else:
                stats, what = self.msg_recv_stats, 'received'
            num_msgs = stats[0]
            if num_msgs == 0:
                continue  # message counts are not recorded by the CkDyn binding
            min_msgsize, max_msgsize, sum_msgsize = stats[1:4]
            avg_msgsize = sum_msgsize / num_msgs
            print('\nMessages ' + what + ': ' + str(num_msgs))
            msgSizeStats = [min_msgsize, avg_msgsize, max_msgsize]
            msgSizeStats = [round(val, 3) for val in msgSizeStats]
            print('Message size in bytes (min / mean / max): ' + ' / '.join([str(v) for v in msgSizeStats]))
            print('Total bytes = ' + str(round(sum_msgsize / 1024.0 / 1024.0, 3)) + ' MB')
        print('')

    def getTopoTreeEdges(self, pe, root_pe, pes=None, bfactor=4):
        """ Returns (parent, children) of 'pe' in a tree spanning the given 'pes',
            or all PEs if 'pes' is None
            If 'pes' is specified, 'root_pe' must be in the first position of 'pes',
            and 'pe' must be a member of 'pes' """
        if pes is not None:
            pes = list(pes)
        return _ckdyn.topo_tree_edges(pe, root_pe, pes, bfactor)

    def getTopoSubtrees(self, root_pe, pes, bfactor=4):
        """ Returns a list of subtrees of root_pe in a spanning tree containing
            all given pes. Subtrees are returned as lists of pes in the
            subtree: the first PE in the list is the root of the subtree, but
            otherwise the list doesn't specify the structure of the subtree
            (the subtree structure can be extracted by recursively calling this
            method). """
        pes = list(pes)
        _, children = _ckdyn.topo_tree_edges(root_pe, root_pe, pes, bfactor)
        subtrees = []
        idx = 1
        num_pes = len(pes)
        for i in range(len(children)):
            if i < len(children) - 1:
                next_child = children[i + 1]
            else:
                next_child = -1
            subtree = []
            while idx < num_pes:
                pe = pes[idx]
                if pe == next_child:
                    break
                subtree.append(pe)
                idx += 1
            subtrees.append(subtree)
        return subtrees

    def myPe(self):
        return self._myPe

    def numPes(self):
        return self._numPes

    def myNode(self):
        """ the logical node (process) of this PE, the index of its nodegroup branches """
        return _ckdyn.my_node()

    def numNodes(self):
        """ the number of logical nodes (processes), the number of branches of a nodegroup """
        return _ckdyn.num_nodes()

    def myHost(self):
        return _ckdyn.physical_node_id(self._myPe)

    def numHosts(self):
        return _ckdyn.num_physical_nodes()

    def getHostPes(self, host):
        return _ckdyn.pes_on_physical_node(host)

    def getHostFirstPe(self, host):
        return _ckdyn.pes_on_physical_node(host)[0]

    def getHostNumPes(self, host):
        return _ckdyn.num_pes_on_physical_node(host)

    def getPeHost(self, pe):
        return _ckdyn.physical_node_id(pe)

    def getPeHostRank(self, pe):
        return _ckdyn.physical_rank(pe)

    def exit(self, exit_code=0):
        _ckdyn.exit(exit_code)

    def abort(self, message):
        _ckdyn.abort(message)

    def addReducer(self, func):
        self.reducers.addReducer(func)

    def LBTurnInstrumentOn(self):
        _ckdyn.lb_instrument(True)

    def LBTurnInstrumentOff(self):
        _ckdyn.lb_instrument(False)

    def myDevice(self):
        """ the GPU device Charm++ (HAPI) assigned to this PE, which is also the
            current CUDA device of the thread running Python code; -1 without
            device support """
        return _ckdyn.my_device()

    def hapiAddCudaCallback(self, stream, future):
        _ckdyn.hapi_add_callback(stream, future)

    #functions for ccs
    def CcsRegisterHandler(self, handlername, handler):
        self.ccs_methods[handlername] = handler
        _ckdyn.ccs_register(handlername)

    def CcsIsRemoteRequest(self):
        return _ckdyn.ccs_is_remote_request()

    def CcsSendReply(self, message):
        _ckdyn.ccs_send_reply(message)

    def CcsDelayReply(self):
        return _ckdyn.ccs_delay_reply()

    def CcsSendDelayedReply(self, d, message):
        _ckdyn.ccs_send_delayed_reply(d, message)

    def callHandler(self, handlername, data):
        try:
            if handlername in self.ccs_methods:
                self.ccs_methods[handlername](data)
            else:
                raise Charm4PyError('Handler ' + handlername + ' not registered')
        except Exception as e:
            self.handleGeneralError(e)


class CharmRemote(Chare):

    def __init__(self):
        charm.thisProxy = self.thisProxy

    def exit(self, exit_code=0):
        charm.exit(exit_code)

    def myPe(self):
        return charm.myPe()

    def LBTurnInstrumentOn(self):
        charm.LBTurnInstrumentOn()

    def LBTurnInstrumentOff(self):
        charm.LBTurnInstrumentOff()

    def addReducer(self, func):
        charm.addReducer(func)

    # user signature is: `def updateGlobals(self, global_dict, module_name='__main__')`
    def updateGlobals(self, *args):
        global_dict = {}
        module_name = args[-1]
        for i in range(0, len(args) - 1, 2):
            global_dict[args[i]] = args[i + 1]

        # TODO remove this warning and related code when the new lb framework is merged
        if charm.myPe() == 0 and charm.lb_requested:
            print('WARNING> updateGlobals with load balancing enabled can lead to unexpected behavior '
                  'due to a bug in Charm++ load balancing. This will be fixed in an upcoming release.')
            charm.lb_requested = False
        sys.modules[module_name].__dict__.update(global_dict)

    def createArray(self, cls, dims=None, ndims=-1, args=[], map=None, useAtSync=False, cb=None):
        proxy = Array(cls, dims, ndims, args, map, useAtSync)
        if cb is not None:
            cb(proxy)
        return proxy

    def rexec(self, code, module_name='__main__'):
        if charm.options.remote_exec is not True:
            raise Charm4PyError('Remote code execution is disabled. Set charm.options.remote_exec to True')
        exec(code, sys.modules[module_name].__dict__)

    def eval(self, expression, module_name='__main__'):
        if charm.options.remote_exec is not True:
            raise Charm4PyError('Remote code execution is disabled. Set charm.options.remote_exec to True')
        return eval(expression, sys.modules[module_name].__dict__)

    def notify_future_deletion(self, store_id, key):
        # a copy of the future that this PE sent under borrow key `key` was deleted
        fut = charm.threadMgr.borrowed_futures[(store_id, key)]
        fut.num_borrowers -= 1
        if fut.num_borrowers == 0:
            # check if threadMgr.futures has the only reference to fid
            # if yes, remove it
            refcount = ctypes.c_long.from_address(id(fut)).value
            # print(store_id, "on pe", charm.myPe(), "depth", depth, "ref count =", refcount)
            if (fut.parent == None and refcount == 3) or (fut.parent != None and refcount == 2):
                # print("Real deletion of", store_id, "from", charm.myPe())
                if fut.parent == None:
                    charm.threadMgr.futures.pop(fut.fid)
                charm.threadMgr.borrowed_futures.pop((store_id, key))

    def propagateException(self, error):
        if time.time() - charm.last_exception_timestamp >= 1.0:
            charm.last_exception_timestamp = time.time()
            if charm.myPe() == 0:
                origin, remote_stacktrace = error.remote_stacktrace
                print('----------------- Python Stack Traceback from PE', origin, '-----------------\n', remote_stacktrace)
                print(type(error).__name__ + ':', error, '(PE ' + str(origin) + ')')
            else:
                self.thisProxy[(charm.myPe()-1) // 2].propagateException(error)

    def printStats(self):
        charm.printStats()

    def _checkpoint_written(self, tag):
        charm._checkpoint_written(tag)

    def registerNewChareType(self, name, source):
        if charm.options.remote_exec is not True:
            raise Charm4PyError('Remote code execution is disabled. Set charm.options.remote_exec to True')
        exec(source, charm.dynamic_register)
        chare_type = charm.dynamic_register[name]
        charm.register(chare_type)
        charm.registerInCharm(chare_type)

    def registerNewChareTypes(self, classes):
        for chare_type in classes:
            charm.register(chare_type)
            charm.registerInCharm(chare_type)


def profile_send_function(func):
    def func_with_profiling(*args, **kwargs):
        em = charm.runningEntryMethod
        if not em.measuringSendTime:
            em.startMeasuringSendTime()
            ret = func(*args, **kwargs)
            em.stopMeasuringSendTime()
        else:
            ret = func(*args, **kwargs)
        return ret
    if hasattr(func, 'ep'):
        func_with_profiling.ep = func.ep
    return func_with_profiling


class __ReadOnlies(object):
    pass


charm = Charm()
readonlies = __ReadOnlies()

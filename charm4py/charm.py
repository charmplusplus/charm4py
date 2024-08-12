#
# @author Juan Galvez (jjgalvez@illinois.edu)
#
# This package allows writing and running Charm++ applications in Python. It
# accesses the C/C++ Charm++ shared library for core runtime functionality,
# and introduces new features thanks to the dynamic language properties of Python.
#
import sys
import os
if sys.version_info < (3, 0, 0):
    import cPickle
    from cStringIO import StringIO
else:
    import pickle as cPickle
    from io import StringIO
import inspect
import time
import gc
import ctypes
from collections import defaultdict
import traceback
from . import chare
from .chare import MAINCHARE, GROUP, ARRAY, CHARM_TYPES
from .chare import CONTRIBUTOR_TYPE_GROUP, CONTRIBUTOR_TYPE_ARRAY
from .chare import Chare, Mainchare, Group, ArrayMap, Array
from . import entry_method
from . import threads
from .threads import Future, LocalFuture, LocalMultiFuture
from . import reduction
from . import wait
from charm4py.c_object_store import MessageBuffer
from . import ray
import array
try:
    import numpy
except ImportError:
    # this is to avoid numpy dependency
    class NumpyDummy:
        ndarray = None
    numpy = NumpyDummy()


def SECTION_ALL(obj):
    return 0


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


# Acts as the Charm runtime at the Python level (there is one instance of this class
# per process)
class Charm(object):

    if os.name == 'nt':
        class PrintStream(object):
            def write(self, msg):
                charm.lib.CkPrintf(msg.encode())
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
        self.arrays = defaultdict(dict)  # aid -> idx -> array element instance with index idx on this PE
        self.entryMethods = {}    # ep_idx -> EntryMethod object
        self.classEntryMethods = [{} for _ in CHARM_TYPES]  # charm_type_id -> class -> list of EntryMethod objects
        self.proxyClasses      = [{} for _ in CHARM_TYPES]  # charm_type_id -> class -> proxy class
        self.groupMsgBuf = defaultdict(list)  # gid -> list of msgs received for constrained groups that haven't been created yet
        self.section_counter = 0
        self.rebuildFuncs = (rebuildByteArray, rebuildArray, rebuildNumpyArray)
        self.sched_tagpool = set(range(1, 128))  # pool of tags for scheduling callables
        self.sched_callables = {}  # tag -> (callable, args)

        self.options = Options()
        self.options.profiling = False
        self.options.pickle_protocol = -1  # -1 selects the highest protocol number
        self.options.local_msg_optim = False
        self.options.local_msg_buf_size = 50
        self.options.auto_flush_wait_queues = True
        self.options.quiet = False
        self.options.remote_exec = False
        self.options.interactive = Options()
        self.options.interactive.verbose = 1
        self.options.interactive.broadcast_imports = True
        self.lib = load_charm_library(self)
        self.ReducerType = self.lib.ReducerType
        self.CkContributeToChare = self.lib.CkContributeToChare
        self.CkContributeToGroup = self.lib.CkContributeToGroup
        self.CkContributeToArray = self.lib.CkContributeToArray
        self.CkContributeToSection = self.lib.CkContributeToSection
        self.CkChareSend = self.lib.CkChareSend
        self.CkGroupSend = self.lib.CkGroupSend
        self.CkArraySend = self.lib.CkArraySend
        self.reducers = reduction.ReducerContainer(self)
        self.redMgr = reduction.ReductionManager(self, self.reducers)
        self.mainchareRegistered = False
        # entry point to Charm program. can be used in place of defining a Mainchare
        self.entry_func = None
        if self.lib.name == 'cython':
            # replace these methods with the fast Cython versions
            self.packMsg = self.lib.packMsg
            self.unpackMsg = self.lib.unpackMsg
        self.interactive = False
        self.last_exception_timestamp = time.time()
        # store chare types defined after program start and other objects created
        # in interactive mode
        self.dynamic_register = sys.modules['__main__'].__dict__
        self.lb_requested = False
        self.threadMgr = threads.EntryMethodThreadManager(self)
        self.createFuture = self.Future = self.threadMgr.createFuture

        # The send buffer is not currently used since we do not buffer messages on
        # the sender. This should be used later when scheduling decisions are to be
        # based on locations of arguments
        self.send_buffer = MessageBuffer()
        self.receive_buffer = MessageBuffer()
        # TODO: maybe implement this buffer in c++
        self.future_get_buffer = {}

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

    def handleGeneralError(self):
        errorType, error, stacktrace = sys.exc_info()
        if not self.interactive:
            if hasattr(error, 'remote_stacktrace'):
                origin, stacktrace = error.remote_stacktrace
                print('----------------- Python Stack Traceback PE ' + str(origin) + ' -----------------')
                print(stacktrace)
            else:
                print('----------------- Python Stack Traceback PE ' + str(self.myPe()) + ' -----------------')
                traceback.print_tb(stacktrace, limit=None)
            self.abort(errorType.__name__ + ': ' + str(error))
        else:
            self.thisProxy[self.myPe()].propagateException(self.prepareExceptionForSend(error))

    def prepareExceptionForSend(self, e):
        if not hasattr(e, 'remote_stacktrace'):
            f = StringIO()
            traceback.print_tb(sys.exc_info()[2], limit=None, file=f)
            e.remote_stacktrace = (self.myPe(), f.getvalue())
        return e

    def process_em_exc(self, e, obj, header):
        if b'block' not in header:
            raise e
        # remote is expecting a response via a future, send exception to the future
        blockFuture = header[b'block']
        sid = None
        if b'sid' in header:
            sid = header[b'sid']
        if b'creation' in header:
            # don't send anything in this case (future is not guaranteed to be used)
            obj.contribute(None, None, blockFuture, sid)
            raise e
        self.prepareExceptionForSend(e)
        if b'bcast' in header:
            if b'bcastret' in header:
                obj.contribute(e, self.reducers.gather, blockFuture, sid)
            else:
                # NOTE: it will work if some elements contribute with an exception (here)
                # and some do nop (None) reduction. Charm++ will ignore the nops
                obj.contribute(e, self.reducers._bcast_exc_reducer, blockFuture, sid)
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

    def buildMainchare(self, onPe, objPtr, ep, args):
        cid = (onPe, objPtr)  # chare ID (objPtr should be a Python int)
        assert onPe == self.myPe()
        assert cid not in self.chares, 'Chare ' + str(cid) + ' already instantiated'
        em = self.entryMethods[ep]
        assert em.name == '__init__', 'Specified mainchare entry method is not constructor'
        self._createInternalChares()
        obj = object.__new__(em.C)  # create object but don't call __init__
        Mainchare.initMember(obj, cid)
        super(em.C, obj).__init__()  # call Chare class __init__ first
        if self.entry_func is not None:
            assert isinstance(obj, chare.DefaultMainchare)
            obj.main = self.entry_func
        if self.options.profiling:
            self.activeChares.add((em.C, Mainchare))
        gc.collect()
        em.run(obj, {}, [args])  # now call the user's __init__
        self.chares[cid] = obj
        if self.myPe() == 0:  # broadcast readonlies
            roData = {}
            for attr in dir(readonlies):  # attr is string
                if not attr.startswith('_') and not attr.endswith('_'):
                    roData[attr] = getattr(readonlies, attr)
            msg = cPickle.dumps(roData, self.options.pickle_protocol)
            # print("Registering readonly data of size " + str(len(msg)))
            self.lib.CkRegisterReadonly(b'charm4py_ro', b'charm4py_ro', msg)
        gc.collect()

    def invokeEntryMethod(self, obj, ep, header, args, ret_fut=False):
        em = self.entryMethods[ep]
        if (em.when_cond is not None) and (not em.when_cond.evaluateWhen(obj, args)):
            obj.__waitEnqueue__(em.when_cond, (0, em, header, args))
        else:
            em.run(obj, header, args, ret_fut=ret_fut)
            if self.options.auto_flush_wait_queues and obj._cond_next is not None:
                obj.__flush_wait_queues__()

    def recvChareMsg(self, chare_id, ep, msg, dcopy_start):
        obj = self.chares[chare_id]
        header, args = self.unpackMsg(msg, dcopy_start, obj)
        self.invokeEntryMethod(obj, ep, header, args)

    def recvGroupMsg(self, gid, ep, msg, dcopy_start):
        if gid in self.groups:
            obj = self.groups[gid]
            header, args = self.unpackMsg(msg, dcopy_start, obj)
            self.invokeEntryMethod(obj, ep, header, args, ret_fut=False)        
        else:
            em = self.entryMethods[ep]
            header, args = self.unpackMsg(msg, dcopy_start, None)
            if em.name != '__init__':
                # this is not a constructor msg and the group hasn't been
                # created yet. this should only happen for constrained groups
                # (buffering of msgs for regular groups that haven't
                # been created yet is done inside Charm++)
                self.groupMsgBuf[gid].append((ep, header, args))
                return
            if b'constrained' in header:
                # constrained group instances are created by SectionManager
                return
            assert gid not in self.groupMsgBuf
            if self.options.profiling:
                self.activeChares.add((em.C, Group))
            obj = object.__new__(em.C)  # create object but don't call __init__
            Group.initMember(obj, gid)
            super(em.C, obj).__init__()  # call Chare class __init__ first
            self.groups[gid] = obj
            em.run(obj, header, args)  # now call the user's __init__

    def arrayMapProcNum(self, gid, index):
        return self.groups[gid].procNum(index)

    def recvArrayMsg(self, aid, index, ep, msg, dcopy_start):
        # print("Array msg received, aid=" + str(aid) + " arrIndex=" + str(index) + " ep=" + str(ep))
        if index in self.arrays[aid]:
            obj = self.arrays[aid][index]
            header, args = self.unpackMsg(msg, dcopy_start, obj)
            dep_ids = []
            is_ray = 'is_ray' in header and header['is_ray']
            if is_ray:
                for i, arg in enumerate(args[:-1]):
                    if isinstance(arg, Future):
                        dep_obj = arg.lookup_object()
                        if dep_obj is None:
                            dep_ids.append(arg.store_id)
                            arg.request_object()
                        else:
                            args[i] = dep_obj
            if len(dep_ids) > 0:
                charm.receive_buffer.insert(dep_ids, (obj, ep, header, args))
            else:
                self.invokeEntryMethod(obj, ep, header, args, ret_fut=is_ray)
        else:
            em = self.entryMethods[ep]
            assert em.name == '__init__', 'Specified array entry method not constructor'
            header, args = self.unpackMsg(msg, dcopy_start, None)
            if self.options.profiling:
                self.activeChares.add((em.C, Array))
            if isinstance(args, Chare):  # obj migrating in
                em = self.entryMethods[ep + 1]  # get 'migrated' EntryMethod object instead of __init__
                obj = args
                obj._contributeInfo = self.lib.initContributeInfo(aid, index, CONTRIBUTOR_TYPE_ARRAY)
                self.arrays[aid][index] = obj
                em.run(obj, {}, ())
            else:
                obj = object.__new__(em.C)   # create object but don't call __init__
                if b'single' in header:
                    Array.initMember(obj, aid, index, single=True)
                else:
                    Array.initMember(obj, aid, index)
                super(em.C, obj).__init__()  # call Chare class __init__ first
                self.arrays[aid][index] = obj
                em.run(obj, header, args)  # now call the user's array element __init__

    def recvArrayBcast(self, aid, indexes, ep, msg, dcopy_start):
        header, args = self.unpackMsg(msg, dcopy_start, None)
        array = self.arrays[aid]
        for index in indexes:
            self.invokeEntryMethod(array[index], ep, header, args)

    def unpackMsg(self, msg, dcopy_start, dest_obj):
        if msg[:7] == b'_local:':
            header, args = dest_obj.__removeLocal__(int(msg[7:]))
        else:
            header, args = cPickle.loads(msg)
            if b'dcopy' in header:
                rel_offset = dcopy_start
                buf = memoryview(msg)
                for arg_pos, typeId, rebuildArgs, size in header[b'dcopy']:
                    arg_buf = buf[rel_offset:rel_offset + size]
                    args[arg_pos] = self.rebuildFuncs[typeId](arg_buf, *rebuildArgs)
                    rel_offset += size
            elif b'custom_reducer' in header:
                reducer = getattr(self.reducers, header[b'custom_reducer'])
                # reduction result won't always be in position 0, but will always be last
                # (e.g. if reduction target is a future, the reduction result will be 2nd argument)
                if reducer.hasPostprocess:
                    args[-1] = reducer.postprocess(args[-1])

        return header, args

    def packMsg(self, destObj, msgArgs, header):
        """Prepares a message for sending, given arguments to an entry method invocation.

          The message is the result of pickling `(header,args)` where header is a dict,
          and args the list of arguments. If direct-copy is enabled, arguments supporting
          the buffer interface will bypass pickling and their place in 'args' will be
          made empty. Instead, metadata to reconstruct these args at the destination will be
          put in the header, and this method will return a list of buffers for
          direct-copying of these args into a CkMessage at Charm side.

          If destination object exists on same PE as source, the args will be stored in
          '_local' buffer of destination obj (without copying), and the msg will be a
          small integer tag to retrieve the args from '_local' when the msg is delivered.

          Args:
              destObj: destination object if it exists on the same PE as source, otherwise None
              msgArgs: arguments to entry method
              header: msg header

          Returns:
              2-tuple containing msg and list of direct-copy buffers

        """
        direct_copy_buffers = []
        dcopy_size = 0
        if destObj is not None:  # if dest obj is local
            localTag = destObj.__addLocal__((header, msgArgs))
            msg = ('_local:' + str(localTag)).encode()
        else:
            direct_copy_hdr = []  # goes to msg header
            args = list(msgArgs)
            if self.lib.direct_copy_supported:
                for i, arg in enumerate(msgArgs):
                    t = type(arg)
                    if t == bytes:
                        nbytes = len(arg)
                        direct_copy_hdr.append((i, 0, (), nbytes))
                    elif t == array.array:
                        nbytes = arg.buffer_info()[1] * arg.itemsize
                        direct_copy_hdr.append((i, 1, (arg.typecode), nbytes))
                    elif t == numpy.ndarray and not arg.dtype.hasobject:
                        # https://docs.scipy.org/doc/numpy/neps/npy-format.html explains what
                        # describes a numpy array
                        # FIXME support Fortran contiguous layout? NOTE that even if we passed
                        # the layout order (array.flags.c_contiguous) to the remote, there is
                        # the issue that when attempting to get a buffer in cffi to the
                        # memoryview, Python throws error: "memoryview: underlying buffer is not
                        # C-contiguous", which seems to be a CPython error (not cffi related)
                        nbytes = arg.nbytes
                        if arg.dtype.isbuiltin:
                            direct_copy_hdr.append((i, 2, (arg.shape, arg.dtype.char), nbytes))
                        else:
                            direct_copy_hdr.append((i, 2, (arg.shape, arg.dtype.name), nbytes))
                    else:
                        continue
                    args[i] = None  # will direct-copy this arg so remove from args list
                    direct_copy_buffers.append(memoryview(arg))
                    dcopy_size += nbytes
                if len(direct_copy_hdr) > 0: header[b'dcopy'] = direct_copy_hdr
            msg = (header, args)
            msg = cPickle.dumps(msg, self.options.pickle_protocol)
        if self.options.profiling:
            self.recordSend(len(msg) + dcopy_size)
        return (msg, direct_copy_buffers)

    # register class C in Charm
    def registerInCharmAs(self, C, charm_type, libRegisterFunc):
        charm_type_id = charm_type.type_id
        entryMethods = self.classEntryMethods[charm_type_id][C]
        # if self.myPe() == 0: print("charm4py:: Registering class " + C.__name__ + " in Charm with " + str(len(entryMethods)) + " entry methods " + str([e.name for e in entryMethods]))
        C.idx[charm_type_id], startEpIdx = libRegisterFunc(C.__name__ + str(charm_type_id), len(entryMethods))
        # if self.myPe() == 0: print("charm4py:: Chare idx=" + str(C.idx[charm_type_id]) + " ctor Idx=" + str(startEpIdx))
        for i, em in enumerate(entryMethods):
            em.epIdx = startEpIdx + i
            self.entryMethods[em.epIdx] = em
        proxyClass = charm_type.__getProxyClass__(C)
        # save proxy class in the same module as its Chare class
        proxyClass.__module__ = C.__module__
        setattr(sys.modules[C.__module__], proxyClass.__name__, proxyClass)
        self.proxyClasses[charm_type_id][C] = proxyClass
        if charm_type_id in (GROUP, ARRAY):
            secProxyClass = charm_type.__getProxyClass__(C, sectionProxy=True)
            secProxyClass.__module__ = C.__module__
            setattr(sys.modules[C.__module__], secProxyClass.__name__, secProxyClass)
            proxyClass.__secproxyclass__ = secProxyClass

    def registerInCharm(self, C):
        C.idx = [None] * len(CHARM_TYPES)
        charm_types = self.registered[C]
        if Mainchare in charm_types:
            self.registerInCharmAs(C, Mainchare, self.lib.CkRegisterMainchare)
        if Group in charm_types:
            if ArrayMap in C.mro():
                self.registerInCharmAs(C, Group, self.lib.CkRegisterArrayMap)
            elif C == SectionManager:
                self.registerInCharmAs(C, Group, self.lib.CkRegisterSectionManager)
            else:
                self.registerInCharmAs(C, Group, self.lib.CkRegisterGroup)
        if Array in charm_types:
            self.registerInCharmAs(C, Array, self.lib.CkRegisterArray)

    # first callback from Charm++ shared library
    # this method registers classes with the shared library
    def registerMainModule(self):
        self._myPe   = self.lib.CkMyPe()
        self._numPes = self.lib.CkNumPes()

        # Charm++ library captures stdout/stderr. here we reset the streams with a buffering
        # policy that ensures that messages reach Charm++ in a timely fashion
        if os.name == 'nt':
            sys.stdout = Charm.PrintStream()
        else:
            sys.stdout = os.fdopen(1, 'wt', 1)
            sys.stderr = os.fdopen(2, 'wt', 1)
        if self.myPe() != 0:
            self.lib.CkRegisterReadonly(b'python_null', b'python_null', None)

        if (self.myPe() == 0) and (not self.options.quiet):
            import platform
            from . import charm4py_version
            py_impl = platform.python_implementation()
            out_msg = ("Charm4py> Running Charm4py version " + charm4py_version +
                       " on Python " + str(platform.python_version()) + " (" +
                       py_impl + "). Using '" +
                       self.lib.name + "' interface to access Charm++")
            if py_impl == 'PyPy':
                if self.lib.name != 'cffi':
                    out_msg += ", **WARNING**: cffi recommended for best performance"
            elif self.lib.name != 'cython':
                out_msg += ", **WARNING**: cython recommended for best performance"
            print(out_msg)
            if sys.version_info < (3,0,0):
                print('\nCharm4py> DEPRECATION: Python 2 support is ending. Some new features may not work.\n')
            if self.options.profiling:
                print('Charm4py> Profiling is ON (this affects performance)')

        for C in self.register_order:
            self.registerInCharm(C)

    def registerAs(self, C, charm_type_id):
        from .sections import SectionManager
        if charm_type_id == MAINCHARE:
            assert not self.mainchareRegistered, 'More than one entry point has been specified'
            self.mainchareRegistered = True
            # make mainchare constructor always a coroutine
            if sys.version_info < (3, 0, 0):
                entry_method.coro(C.__init__.im_func)
            else:
                entry_method.coro(C.__init__)
        charm_type = chare.charm_type_id_to_class[charm_type_id]
        # print("charm4py: Registering class " + C.__name__, "as", charm_type.__name__, "type_id=", charm_type_id, charm_type)
        profilingOn = self.options.profiling
        ems = [entry_method.EntryMethod(C, m, profilingOn) for m in charm_type.__baseEntryMethods__()]

        members = dir(C)
        if C == SectionManager:
            ems.append(entry_method.EntryMethod(C, 'sendToSection', profilingOn))
            members.remove('sendToSection')
        self.classEntryMethods[charm_type_id][C] = ems

        for m in members:
            m_obj = getattr(C, m)
            if not callable(m_obj) or inspect.isclass(m_obj):
                continue
            if m in chare.method_restrictions['reserved'] and m_obj != getattr(Chare, m):
                raise Charm4PyError(str(C) + " redefines reserved method '"  + m + "'")
            if m.startswith('__') and m.endswith('__'):
                continue  # filter out non-user methods
            if m in chare.method_restrictions['non_entry_method']:
                continue
            if charm_type_id != ARRAY and m in {'migrate', 'setMigratable'}:
                continue
            # print(m)
            em = entry_method.EntryMethod(C, m, profilingOn)
            self.classEntryMethods[charm_type_id][C].append(em)
        self.registered[C].add(charm_type)

    # called by user (from Python) to register their Charm++ classes with the Charm4py runtime
    # by default a class is registered to work with both Groups and Arrays
    def register(self, C, collections=(GROUP, ARRAY)):
        if C in self.registered:
            return
        if (not hasattr(C, 'mro')) or (Chare not in C.mro()):
            raise Charm4PyError('Only subclasses of Chare can be registered')

        # cache of template condition objects for `chare.wait(cond_str)` calls
        # maps cond_str to condition object. the condition object stores the lambda function associated with cond_str
        # TODO: remove old/unused condition strings from dict?
        C.__charm_wait_conds__ = {}
        self.registered[C] = set()
        for charm_type_id in collections:
            self.registerAs(C, charm_type_id)
        self.register_order.append(C)

    def _registerInternalChares(self):
        global SectionManager
        from .sections import SectionManager
        self.register(SectionManager, (GROUP,))

        self.register(CharmRemote, (GROUP,))

        from .pool import PoolScheduler, Worker
        if self.interactive:
            if sys.version_info < (3, 0, 0):
                entry_method.coro(PoolScheduler.start.im_func)
                entry_method.coro(PoolScheduler.startSingleTask.im_func)
            else:
                entry_method.coro(PoolScheduler.start)
                entry_method.coro(PoolScheduler.startSingleTask)
        self.register(PoolScheduler, (ARRAY,))
        self.register(Worker, (GROUP,))

        if self.options.profiling:
            self.internalChareTypes.update({SectionManager, CharmRemote,
                                            PoolScheduler, Worker})

    def _createInternalChares(self):
        Group(CharmRemote)
        Group(SectionManager)

        from .pool import Pool, PoolScheduler
        pool_proxy = Chare(PoolScheduler, onPE=0)
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
        self.lib.start()

    def arrayElemLeave(self, aid, index):
        obj = self.arrays[aid].pop(index)
        if hasattr(obj, '_scookies'):
            charm.abort('Cannot migrate elements that are part of a section '
                        '(this will be supported in a future version)')
        self.threadMgr.objMigrating(obj)
        if hasattr(obj, '__channels__'):
            assert len(obj.__pendingChannels__) == 0, 'Cannot migrate chares that did not complete channel establishment'
        del obj._contributeInfo  # don't want to pickle this
        pickled_chare = cPickle.dumps(({}, obj), self.options.pickle_protocol)
        # facilitate garbage collection (especially by removing cyclical references)
        del obj._local
        del obj._local_free_head
        del obj._active_grp_conds
        obj._cond_next = None
        obj._cond_last = None
        return pickled_chare

    # Charm class contribute function used by Array, Group and Sections for reductions
    # 'section' can either be an sid (2-tuple) or a section proxy
    def contribute(self, data, reducer, target, chare, section=None):
        if section is None and not chare.thisProxy.issec:
            contribution = self.redMgr.prepare(data, reducer, chare)
            fid = 0
            if isinstance(target, Future):
                fid = target.fid
                target = target.getTargetProxyEntryMethod()
            contributeInfo = self.lib.getContributeInfo(target.ep, fid, contribution, chare)
            if self.options.profiling:
                self.recordSend(contributeInfo.getDataSize())
            target.__self__.ckContribute(contributeInfo)
        else:
            if section is None:
                # for constrained groups, thisProxy is a section proxy
                sid = chare.thisProxy.section[1]  # get the sid from the proxy
            elif isinstance(section, tuple):
                sid = section  # already a sid
            else:
                # is a section proxy
                sid = section.section[1]
            if isinstance(reducer, tuple):
                reducer = reducer[1]
            if reducer is not None and reducer.hasPreprocess:
                data = reducer.preprocess(data, chare)
            try:
                redno = chare._scookies[sid]
            except:
                raise Charm4PyError('Chare doing section reduction but is not part of a section')
            self.sectionMgr.contrib(sid, redno, data, reducer, target)
            chare._scookies[sid] += 1

    def combine(self, *proxies):
        sid = (self._myPe, self.section_counter)
        self.section_counter += 1
        pes = set()
        futures = [self.Future() for _ in range(len(proxies))]
        for i, proxy in enumerate(proxies):
            secproxy = None
            if proxy.issec:
                secproxy = proxy
            proxy._getSectionLocations_(sid, 1, SECTION_ALL, None, None, futures[i], secproxy)
        for f in futures:
            pes.update(f.get()[0])
        assert len(pes) > 0
        root = min(pes)
        self.sectionMgr.thisProxy[root].createSectionDown(sid, pes, None)
        return proxies[0].__getsecproxy__((root, sid))

    def split(self, proxy, numsections, section_func=None, elems=None, slicing=None, cons=None):
        assert (hasattr(proxy, 'gid') and proxy.elemIdx == -1) or (hasattr(proxy, 'aid') and proxy.elemIdx == ())
        sid0 = (self._myPe, self.section_counter)
        self.section_counter += numsections
        secproxy = None
        if proxy.issec:
            secproxy = proxy
        if elems is None:
            f = self.Future()
            proxy._getSectionLocations_(sid0, numsections, section_func, slicing, None, f, secproxy)
            section_pes = f.get()
        else:
            if numsections == 1 and not isinstance(elems[0], list) and not isinstance(elems[0], set):
                elems = [elems]
            try:
                assert len(elems) == numsections
            except AssertionError:
                print(len(elems), numsections)
            if hasattr(proxy, 'gid') and not proxy.issec:
                # in this case the elements are guaranteed to be PEs, so I don't
                # have to collect locations
                section_pes = elems
            else:
                f = self.Future()
                proxy._getSectionLocations_(sid0, numsections, None, None, elems, f, secproxy)
                section_pes = f.get()
        secProxies = []
        # TODO if there are many many sections, should do a stateless multicast to the roots with the section info
        for i in range(numsections):
            sid = (self._myPe, sid0[1] + i)
            pes = section_pes[i]
            if not isinstance(pes, set):
                # pes will be a set in most cases, unless the user passed a list of elements.
                # transforming to set ensures we get rid of duplicates
                pes = set(pes)
            assert len(pes) > 0
            root = min(pes)
            if not proxy.issec and hasattr(proxy, 'gid'):
                self.sectionMgr.thisProxy[root].createGroupSectionDown(sid, proxy.gid, pes, None, cons)
            else:
                self.sectionMgr.thisProxy[root].createSectionDown(sid, pes, None)
            secProxies.append(proxy.__getsecproxy__((root, sid)))
        return secProxies

    def startQD(self, callback):
        fid = 0
        if isinstance(callback, Future):
            fid = callback.fid
            callback = callback.getTargetProxyEntryMethod()
        cb_proxy = callback.__self__
        if hasattr(cb_proxy, 'section'):
            self.lib.CkStartQD_SectionCallback(cb_proxy.section[1], cb_proxy.section[0], callback.ep)
        elif hasattr(cb_proxy, 'gid'):
            self.lib.CkStartQD_GroupCallback(cb_proxy.gid, cb_proxy.elemIdx, callback.ep, fid)
        elif hasattr(cb_proxy, 'aid'):
            self.lib.CkStartQD_ArrayCallback(cb_proxy.aid, cb_proxy.elemIdx, callback.ep, fid)
        else:
            self.lib.CkStartQD_ChareCallback(cb_proxy.cid, callback.ep, fid)

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
        tag = self.sched_tagpool.pop()
        self.sched_callables[tag] = (callable_obj, args)
        self.lib.scheduleTagAfter(tag, secs * 1000)

    def triggerCallable(self, tag):
        if self.options.profiling:
            self.triggerCallableEM.startMeasuringTime()
        cb, args = self.sched_callables.pop(tag)
        self.sched_tagpool.add(tag)
        cb(*args)
        if self.options.profiling:
            self.triggerCallableEM.stopMeasuringTime()

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

        print('Timings for PE', self.myPe(), ':')
        table = [['', 'em', 'send', 'recv', 'total']]
        lineNb = 1
        sep = {}
        row_totals = [0.0] * 4
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
                vals = em.times + [sum(em.times)]
                for i in range(len(row_totals)):
                    row_totals[i] += vals[i]
                table.append([em_name] + [str(round(v, 3)) for v in vals])
                lineNb += 1
        sep[lineNb] = '-----------------------------------------------------------'
        table.append([''] + [str(round(v, 3)) for v in row_totals])
        lineNb += 1
        sep[lineNb] = '-----------------------------------------------------------'
        misc_overheads = [str(round(v, 3)) for v in self.lib.times]
        table.append(['reductions', ' ', ' ', misc_overheads[0], misc_overheads[0]])
        table.append(['custom reductions',   ' ', ' ', misc_overheads[1], misc_overheads[1]])
        table.append(['migrating out',  ' ', ' ', misc_overheads[2], misc_overheads[2]])
        lineNb += 3
        triggerCallableTotalTime = sum(self.triggerCallableEM.times)
        if triggerCallableTotalTime > 0:
            vals = self.triggerCallableEM.times + [triggerCallableTotalTime]
            for i, v in enumerate(vals):
                row_totals[i] += v
            times = [str(round(v, 3)) for v in vals]
            table.append(['triggerCallable'] + times)
            lineNb += 1
        sep[lineNb] = '-----------------------------------------------------------'
        row_totals[2] += sum(self.lib.times)
        row_totals[3] += sum(self.lib.times)
        table.append([''] + [str(round(v, 3)) for v in row_totals])
        lineNb += 1
        self.__printTable__(table, sep)

        for i in (0, 1):
            if i == 0:
                num_msgs = self.msg_send_stats[0]
                min_msgsize, max_msgsize, sum_msgsize = self.msg_send_stats[1:4]
                avg_msgsize = sum_msgsize / num_msgs
                print('\nMessages sent: ' + str(num_msgs))
            else:
                num_msgs = self.msg_recv_stats[0]
                min_msgsize, max_msgsize, sum_msgsize = self.msg_recv_stats[1:4]
                avg_msgsize = sum_msgsize / num_msgs
                print('\nMessages received: ' + str(num_msgs))
            msgSizeStats = [min_msgsize, avg_msgsize, max_msgsize]
            msgSizeStats = [round(val, 3) for val in msgSizeStats]
            print('Message size in bytes (min / mean / max): ' + ' / '.join([str(v) for v in msgSizeStats]))
            print('Total bytes = ' + str(round(sum_msgsize / 1024.0 / 1024.0, 3)) + ' MB')
        print('')

    def lib_version_check(self, commit_id_str):
        req_version = tuple([int(n) for n in open(os.path.dirname(__file__) + '/libcharm_version', 'r').read().split('.')])
        version = [int(n) for n in commit_id_str.split('-')[0][1:].split('.')]
        try:
            version = tuple(version + [int(commit_id_str.split('-')[1])])
        except:
            version = tuple(version + [0])
        if version < req_version:
            req_str = '.'.join([str(n) for n in req_version])
            cur_str = '.'.join([str(n) for n in version])
            raise Charm4PyError('Charm++ version >= ' + req_str + ' required. ' +
                                'Existing version is ' + cur_str)

    def getTopoTreeEdges(self, pe, root_pe, pes=None, bfactor=4):
        """ Returns (parent, children) of 'pe' in a tree spanning the given 'pes',
            or all PEs if 'pes' is None
            If 'pes' is specified, 'root_pe' must be in the first position of 'pes',
            and 'pe' must be a member of 'pes' """
        return self.lib.getTopoTreeEdges(pe, root_pe, pes, bfactor)

    def getTopoSubtrees(self, root_pe, pes, bfactor=4):
        """ Returns a list of subtrees of root_pe in a spanning tree containing
            all given pes. Subtrees are returned as lists of pes in the
            subtree: the first PE in the list is the root of the subtree, but
            otherwise the list doesn't specify the structure of the subtree
            (the subtree structure can be extracted by recursively calling this
            method). """
        return self.lib.getTopoSubtrees(root_pe, pes, bfactor)

    def myPe(self):
        return self._myPe

    def numPes(self):
        return self._numPes

    def myHost(self):
        return self.lib.CkPhysicalNodeID(self._myPe)

    def numHosts(self):
        return self.lib.CkNumPhysicalNodes()

    def getHostPes(self, host):
        return self.lib.CkGetPesOnPhysicalNode(host)

    def getHostFirstPe(self, host):
        return self.lib.CkGetFirstPeOnPhysicalNode(host)

    def getHostNumPes(self, host):
        return self.lib.CkNumPesOnPhysicalNode(host)

    def getPeHost(self, pe):
        return self.lib.CkPhysicalNodeID(pe)

    def getPeHostRank(self, pe):
        return self.lib.CkPhysicalRank(pe)

    def exit(self, exit_code=0):
        self.lib.CkExit(exit_code)

    def abort(self, message):
        self.lib.CkAbort(message)

    def addReducer(self, func):
        self.reducers.addReducer(func)

    def LBTurnInstrumentOn(self):
        self.lib.LBTurnInstrumentOn()

    def LBTurnInstrumentOff(self):
        self.lib.LBTurnInstrumentOff()


class CharmRemote(Chare):

    def __init__(self):
        charm.thisProxy = self.thisProxy

    def exit(self, exit_code=0):
        charm.exit(exit_code)

    def myPe(self):
        return charm.myPe()

    def LBTurnInstrumentOn(self):
        charm.lib.LBTurnInstrumentOn()

    def LBTurnInstrumentOff(self):
        charm.lib.LBTurnInstrumentOff()

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

    # deposit value of one of the futures that was created on this PE
    def _future_deposit_result(self, fid, result=None):
        charm.threadMgr.depositFuture(fid, result)

    def notify_future_deletion(self, store_id, depth):
        charm.threadMgr.borrowed_futures[(store_id, depth)].num_borrowers -= 1
        if charm.threadMgr.borrowed_futures[(store_id, depth)].num_borrowers == 0:
            # check if threadMgr.futures has the only reference to fid
            # if yes, remove it
            fut = charm.threadMgr.borrowed_futures[(store_id, depth)]
            refcount = ctypes.c_long.from_address(id(fut)).value
            #print(store_id, "on pe", charm.myPe(), "depth", depth, "ref count =", refcount)
            if (fut.parent == None and refcount == 3) or (fut.parent != None and refcount == 2):
                #print("Real deletion of", store_id, "from", charm.myPe())
                if fut.parent == None:
                    charm.threadMgr.futures.pop(fut.fid)
                charm.threadMgr.borrowed_futures.pop((store_id, depth))

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


def load_charm_library(charm):
    args = sys.argv
    libcharm_path = os.path.join(os.path.dirname(__file__), '.libs')
    if os.name == 'nt':
        os.environ['PATH'] += ';' + libcharm_path
    if '+libcharm_interface' in args:
        arg_idx = args.index('+libcharm_interface')
        interface = args.pop(arg_idx + 1)
        args.pop(arg_idx)
        if interface == 'ctypes':
            from .charmlib.charmlib_ctypes import CharmLib
        elif interface == 'cffi':
            from .charmlib.charmlib_cffi import CharmLib
        elif interface == 'cython':
            from .charmlib.charmlib_cython import CharmLib
        else:
            raise Charm4PyError('Unrecognized interface ' + interface)
    else:
        # pick best available interface
        import platform
        py_impl = platform.python_implementation()
        if py_impl != 'PyPy':
            try:
                from .charmlib.charmlib_cython import CharmLib
            except:
                try:
                    from .charmlib.charmlib_cffi import CharmLib
                except:
                    from .charmlib.charmlib_ctypes import CharmLib
        else:
            # for PyPy we require the cffi interface (cffi comes builtin in PyPy)
            from .charmlib.charmlib_cffi import CharmLib
    return CharmLib(charm, charm.options, libcharm_path)


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


def rebuildByteArray(data):
    return bytes(data)


def rebuildArray(data, typecode):
    #a = array.array('d', data.cast(typecode))  # this is slow
    a = array.array(typecode)
    a.frombytes(data)
    return a


def rebuildNumpyArray(data, shape, dt):
    a = numpy.frombuffer(data, dtype=numpy.dtype(dt))  # this does not copy
    a.shape = shape
    return a.copy()


charm = Charm()
readonlies = __ReadOnlies()

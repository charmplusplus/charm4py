#
# @author Juan Galvez (jjgalvez@illinois.edu)
#
# This package allows writing and running Charm++ applications in Python. It
# accesses the C/C++ Charm++ shared library for core runtime functionality.
#
import sys
import os
if sys.version_info < (3, 0, 0):
    import cPickle
else:
    import pickle as cPickle
import time
from collections import defaultdict
from . import chare
from .chare import MAINCHARE, GROUP, ARRAY, CHARM_TYPES
from .chare import CONTRIBUTOR_TYPE_GROUP, CONTRIBUTOR_TYPE_ARRAY
from .chare import Chare, Mainchare, Group, ArrayMap, Array
from . import entry_method
from . import threads
from .threads import Future
from . import reduction
from . import wait
import array
try:
    import numpy
except ImportError:
    # this is to avoid numpy dependency
    class NumpyDummy:
        ndarray = None
    numpy = NumpyDummy()


class Options:
    PROFILING = False
    PICKLE_PROTOCOL = -1    # -1 is highest protocol number
    LOCAL_MSG_OPTIM = True
    LOCAL_MSG_BUF_SIZE = 50
    AUTO_FLUSH_WAIT_QUEUES = True
    QUIET = False


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
        self.msg_send_sizes = []  # for profiling
        self.msg_recv_sizes = []  # for profiling
        self.runningEntryMethod = None  # currently running entry method (only used with profiling)
        # track chare constructor call stack that occurs in mainchare due to
        # inlining of constructor calls (only used with profiling)
        self.mainchareEmStack = []
        self.activeChares = set()  # for profiling (active chares on this PE)
        self.opts = Options
        self.rebuildFuncs = [rebuildByteArray, rebuildArray, rebuildNumpyArray]
        self.lib = load_charm_library(self)
        self.ReducerType = self.lib.ReducerType
        self.CkContributeToChare = self.lib.CkContributeToChare
        self.CkContributeToGroup = self.lib.CkContributeToGroup
        self.CkContributeToArray = self.lib.CkContributeToArray
        self.CkChareSend = self.lib.CkChareSend
        self.CkGroupSend = self.lib.CkGroupSend
        self.CkArraySend = self.lib.CkArraySend
        self.reducers = reduction.ReducerContainer(self)
        self.redMgr = reduction.ReductionManager(self, self.reducers)
        self.mainchareRegistered = False
        # entry point to Charm program. can be used in place of defining a Mainchare
        self.entry_func = None
        if self.lib.name == 'cython':
            # replace these methods with the fast cython versions
            self.packMsg = self.lib.packMsg
            self.unpackMsg = self.lib.unpackMsg
        # cache of template condition objects for `chare.wait(cond_str)` calls
        # maps cond_str to condition object. the condition object stores the lambda function associated with cond_str
        # TODO: remove old/unused condition strings
        self.wait_conditions = {}
        # store chare types defined after program start and other objects created
        # in interactive mode
        self.dynamic_register = {}

    def handleGeneralError(self):
        import traceback
        errorType, error, stacktrace = sys.exc_info()
        print("----------------- Python Stack Traceback PE " + str(self.myPe()) + " -----------------")
        traceback.print_tb(stacktrace, limit=None)
        self.abort(errorType.__name__ + ": " + str(error))

    def recvReadOnly(self, msg):
        roData = cPickle.loads(msg)
        for name, obj in roData.items():
            setattr(readonlies, name, obj)

    def buildMainchare(self, onPe, objPtr, ep, args):
        cid = (onPe, objPtr)  # chare ID (objPtr should be a Python int)
        assert onPe == self.myPe()
        assert cid not in self.chares, "Chare " + str(cid) + " already instantiated"
        em = self.entryMethods[ep]
        assert em.isCtor, "Specified mainchare entry method is not constructor"
        self._createInternalChares()
        obj = object.__new__(em.C)  # create object but don't call __init__
        Mainchare.initMember(obj, cid)
        super(em.C, obj).__init__()  # call Chare class __init__ first
        if self.entry_func is not None:
            assert isinstance(obj, chare.DefaultMainchare)
            obj.main = self.entry_func
            del self.entry_func
        if Options.PROFILING:
            self.activeChares.add((em.C, Mainchare))
            em.startMeasuringTime()
        self.threadMgr.startThread(obj, em, [args], {})  # call user's __init__ in a new thread
        self.chares[cid] = obj
        if Options.PROFILING:
            em.stopMeasuringTime()
        if self.myPe() == 0:  # broadcast readonlies
            roData = {}
            for attr in dir(readonlies):  # attr is string
                if not attr.startswith("_") and not attr.endswith("_"):
                    roData[attr] = getattr(readonlies, attr)
            msg = cPickle.dumps(roData, Options.PICKLE_PROTOCOL)
            # print("Registering readonly data of size " + str(len(msg)))
            self.lib.CkRegisterReadonly(b"charm4py_ro", b"charm4py_ro", msg)

    def invokeEntryMethod(self, obj, ep, header, args, t0):
        em = self.entryMethods[ep]
        if Options.PROFILING:
            em.addRecvTime(time.time() - t0)
            em.startMeasuringTime()

        if (em.when_cond is not None) and (not em.when_cond.evaluateWhen(obj, args)):
            obj.__waitEnqueue__(em.when_cond, (0, em, header, args))
        else:
            self.mainThreadEntryMethod = em
            em.run(obj, header, args)
            if Options.AUTO_FLUSH_WAIT_QUEUES and obj._cond_next is not None:
                obj.__flush_wait_queues__()

        if Options.PROFILING:
            em.stopMeasuringTime()

    def recvChareMsg(self, chare_id, ep, msg, t0, dcopy_start):
        obj = self.chares[chare_id]
        header, args = self.unpackMsg(msg, dcopy_start, obj)
        self.invokeEntryMethod(obj, ep, header, args, t0)

    def recvGroupMsg(self, gid, ep, msg, t0, dcopy_start):
        if gid in self.groups:
            obj = self.groups[gid]
            header, args = self.unpackMsg(msg, dcopy_start, obj)
            self.invokeEntryMethod(obj, ep, header, args, t0)
        else:
            em = self.entryMethods[ep]
            assert em.isCtor, "Specified group entry method not constructor"
            header, args = self.unpackMsg(msg, dcopy_start, None)
            obj = object.__new__(em.C)  # create object but don't call __init__
            Group.initMember(obj, gid)
            super(em.C, obj).__init__()  # call Chare class __init__ first
            if Options.PROFILING:
                self.activeChares.add((em.C, Group))
                if self.runningEntryMethod is not None:
                    self.mainchareEmStack.append(self.runningEntryMethod)
                    self.runningEntryMethod.stopMeasuringTime()
                em.addRecvTime(time.time() - t0)
                em.startMeasuringTime()
            obj.__init__(*args)          # now call the user's __init__
            self.groups[gid] = obj
            if b'block' in header:
                obj.contribute(None, None, header[b'block'])
            if Options.PROFILING:
                em.stopMeasuringTime()
                if len(self.mainchareEmStack) > 0:
                    self.mainchareEmStack.pop().startMeasuringTime()

    def arrayMapProcNum(self, gid, index):
        return self.groups[gid].procNum(index)

    def recvArrayMsg(self, aid, index, ep, msg, t0, dcopy_start):
        # print("Array msg received, aid=" + str(aid) + " arrIndex=" + str(index) + " ep=" + str(ep))
        if index in self.arrays[aid]:
            obj = self.arrays[aid][index]
            header, args = self.unpackMsg(msg, dcopy_start, obj)
            self.invokeEntryMethod(obj, ep, header, args, t0)
        else:
            em = self.entryMethods[ep]
            assert em.isCtor, "Specified array entry method not constructor"
            header, args = self.unpackMsg(msg, dcopy_start, None)
            if Options.PROFILING:
                self.activeChares.add((em.C, Array))
                em.addRecvTime(time.time() - t0)
            if isinstance(args, Chare):  # obj migrating in
                obj = args
                obj._contributeInfo = self.lib.initContributeInfo(aid, index, CONTRIBUTOR_TYPE_ARRAY)
            else:
                obj = object.__new__(em.C)   # create object but don't call __init__
                Array.initMember(obj, aid, index)
                super(em.C, obj).__init__()  # call Chare class __init__ first
                if Options.PROFILING:
                    if self.runningEntryMethod is not None:
                        self.mainchareEmStack.append(self.runningEntryMethod)
                        self.runningEntryMethod.stopMeasuringTime()
                    em.startMeasuringTime()
                obj.__init__(*args)          # now call the user's array element __init__
                if b'block' in header:
                    obj.contribute(None, None, header[b'block'])
                if Options.PROFILING:
                    em.stopMeasuringTime()
                    if len(self.mainchareEmStack) > 0:
                        self.mainchareEmStack.pop().startMeasuringTime()
            self.arrays[aid][index] = obj

    def unpackMsg(self, msg, dcopy_start, dest_obj):
        if msg[:7] == b"_local:":
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
            elif b"custom_reducer" in header:
                reducer = getattr(self.reducers, header[b"custom_reducer"])
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
            msg = ("_local:" + str(localTag)).encode()
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
                        direct_copy_hdr.append((i, 2, (arg.shape, arg.dtype.name), nbytes))
                    else:
                        continue
                    args[i] = None  # will direct-copy this arg so remove from args list
                    direct_copy_buffers.append(memoryview(arg))
                    dcopy_size += nbytes
                if len(direct_copy_hdr) > 0: header[b'dcopy'] = direct_copy_hdr
            msg = (header, args)
            msg = cPickle.dumps(msg, Options.PICKLE_PROTOCOL)
        if Options.PROFILING: self.recordSend(len(msg) + dcopy_size)
        return (msg, direct_copy_buffers)

    # register class C in Charm
    def registerInCharmAs(self, C, charm_type, libRegisterFunc):
        charm_type_id = charm_type.type_id
        entryMethods = self.classEntryMethods[charm_type_id][C]
        # if self.myPe() == 0: print("charm4py:: Registering class " + C.__name__ + " in Charm with " + str(len(entryMethods)) + " entry methods " + str([e.name for e in entryMethods]))
        C.idx[charm_type_id], startEpIdx = libRegisterFunc(C.__name__ + str(charm_type_id), len(entryMethods))
        # if self.myPe() == 0: print("charm4py:: Chare idx=" + str(C.idx[charm_type_id]) + " ctor Idx=" + str(startEpIdx))
        for i, em in enumerate(entryMethods):
            if i == 0:
                em.isCtor = True
            em.epIdx = startEpIdx + i
            self.entryMethods[em.epIdx] = em
        proxyClass = charm_type.__getProxyClass__(C)
        self.proxyClasses[charm_type_id][C] = proxyClass
        setattr(self, proxyClass.__name__, proxyClass)   # save new class in my namespace
        setattr(chare, proxyClass.__name__, proxyClass)  # save in module namespace (needed to pickle the proxy)

    def registerInCharm(self, C):
        C.idx = [None] * len(CHARM_TYPES)
        charm_types = self.registered[C]
        if Mainchare in charm_types:
            self.registerInCharmAs(C, Mainchare, self.lib.CkRegisterMainchare)
        if Group in charm_types:
            if ArrayMap in C.mro():
                self.registerInCharmAs(C, Group, self.lib.CkRegisterArrayMap)
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
            self.lib.CkRegisterReadonly(b"python_null", b"python_null", None)

        if (self.myPe() == 0) and (not Options.QUIET):
            import platform
            from . import charm4py_version
            out_msg = ("charm4py> Running Charm4py version " + charm4py_version +
                       " on Python " + str(platform.python_version()) + " (" +
                       str(platform.python_implementation()) + "). Using '" +
                       self.lib.name + "' interface to access Charm++")
            if self.lib.name != "cython":
                out_msg += ", **WARNING**: cython recommended for best performance"
            print(out_msg)

        for C in self.register_order:
            self.registerInCharm(C)

    def registerAs(self, C, charm_type_id):
        if charm_type_id == MAINCHARE:
            assert not self.mainchareRegistered, "More than one entry point has been specified"
            self.mainchareRegistered = True
        charm_type = chare.charm_type_id_to_class[charm_type_id]
        # print("charm4py: Registering class " + C.__name__, "as", charm_type.__name__, "type_id=", charm_type_id, charm_type)
        l = [entry_method.EntryMethod(C, m, profile=Options.PROFILING)
                                     for m in charm_type.__baseEntryMethods__()]
        self.classEntryMethods[charm_type_id][C] = l
        for m in dir(C):
            if not callable(getattr(C, m)):
                continue
            if m in chare.method_restrictions['reserved'] and getattr(C, m) != getattr(Chare, m):
                raise Charm4PyError(str(C) + " redefines reserved method '"  + m + "'")
            if m.startswith("__") and m.endswith("__"):
                continue  # filter out non-user methods
            if m in chare.method_restrictions['non_entry_method']:
                continue
            # print(m)
            em = entry_method.EntryMethod(C, m, profile=Options.PROFILING)
            self.classEntryMethods[charm_type_id][C].append(em)
        self.registered[C].add(charm_type)

    # called by user (from Python) to register their Charm++ classes with the charm4py runtime
    # by default a class is registered to work with both Groups and Arrays
    def register(self, C, collections=(GROUP, ARRAY)):
        if C in self.registered:
            return
        if (not hasattr(C, 'mro')) or (Chare not in C.mro()):
            raise Charm4PyError("Only subclasses of Chare can be registered")

        self.registered[C] = set()
        for charm_type_id in collections:
            self.registerAs(C, charm_type_id)
        self.register_order.append(C)

    def _registerInternalChares(self):
        self.register(CharmRemote, (GROUP,))

    def _createInternalChares(self):
        Group(CharmRemote)

    def start(self, entry=None, classes=[], modules=[], interactive=False):
        """
        Start charm4py program.

        IMPORTANT: classes must be registered in the same order on all processes. In
        other words, the arguments to this method must have the same ordering on all
        processes.

        Args:
            entry:   program entry point (function or Chare class)
            classes: list of Charm classes to register with runtime
            modules: list of names of modules containing Charm classes (all of the Charm
                     classes defined in the module will be registered). method will
                     always search module '__main__' for Charm classes even if no
                     arguments are passed to this method.
        """

        if interactive:
            from .interactive import InteractiveConsole as entry
            self.origStdinFd = os.dup(0)
            self.origStoutFd = os.dup(1)
            self.dynamic_register.update({'charm': charm, 'Chare': Chare, 'Group': Group,
                                          'Array': Array, 'Reducer': self.reducers})

        if self.started:
            raise Charm4PyError("charm.start() can only be called once")
        self.started = True

        if Options.PROFILING:
            self.contribute = profile_send_function(self.contribute)
        if "++quiet" in sys.argv:
            Options.QUIET = True

        self._registerInternalChares()

        if hasattr(entry, 'mro') and Chare in entry.mro():
            if entry.__init__.__code__.co_argcount != 2:
                raise Charm4PyError("Mainchare constructor must have only one parameter")
            self.register(entry, (MAINCHARE,))
        else:
            assert callable(entry), "Given entry point is not a function or Chare"
            if entry.__code__.co_argcount != 1:
                raise Charm4PyError("Main function must have only one parameter")
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
        import inspect
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
                        raise Charm4PyError("Chares must not inherit from Group, Array or"
                                           " Mainchare. Refer to new API")

        for module in (chare, entry_method, threads, wait):
            module.charmStarting()
        self.threadMgr = threads.EntryMethodThreadManager()
        self.createFuture = self.threadMgr.createFuture

        self.lib.start()

    def arrayElemLeave(self, aid, index):
        obj = self.arrays[aid].pop(index)
        self.threadMgr.objMigrating(obj)
        del obj._contributeInfo  # don't want to pickle this
        return cPickle.dumps(({}, obj), Options.PICKLE_PROTOCOL)

    # Charm class level contribute function used by Array, Group for reductions
    def contribute(self, data, reducer, target, contributor):
        contribution = self.redMgr.prepare(data, reducer, contributor)
        fid = 0
        if isinstance(target, Future):
            fid = target.fid
            proxy_class = getattr(self, target.proxy_class_name)
            proxy = proxy_class.__new__(proxy_class)
            proxy.__setstate__(target.proxy_state)
            target = proxy._future_deposit_result
        contributeInfo = self.lib.getContributeInfo(target.ep, fid, contribution, contributor)
        if Options.PROFILING:
            self.recordSend(contributeInfo.getDataSize())
        target.__self__.ckContribute(contributeInfo)

    def awaitCreation(self, *proxies):
        for proxy in proxies:
            if not hasattr(proxy, 'creation_future'):
                if not proxy.__class__.__name__.endswith("Proxy"):
                    raise Charm4PyError('Did not pass a proxy to awaitCreation? ' + str(type(proxy)))
                raise Charm4PyError('awaitCreation can only be used if creation triggered from threaded entry method')
            proxy.creation_future.get()
            del proxy.creation_future

    def recordSend(self, size):
        # TODO? might be better (certainly more memory efficient) to update msg stats
        # like min, max, total, each time a send is recorded instead of storing the msg
        # size of all sent messages and calculating stats at the end. same applies to receives
        self.msg_send_sizes.append(size)

    def recordReceive(self, size):
        self.msg_recv_sizes.append(size)

    def __printTable__(self, table, sep):
        col_width = [max(len(x) for x in col) for col in zip(*table)]
        for j, line in enumerate(table):
            if j in sep: print(sep[j])
            print("| " + " | ".join("{:{}}".format(x, col_width[i]) for i, x in enumerate(line)) + " |")

    def printStats(self):
        if not self.started:
            raise Charm4PyError('charm was not started')
        if not Options.PROFILING:
            print('NOTE: called charm.printStats() but profiling is disabled')
            return

        if self.runningEntryMethod is not None:
            # record elapsed time of current entry method so that it is part of displayed stats
            em = self.runningEntryMethod
            assert not em.measuringSendTime
            em.stopMeasuringTime()
            em.startMeasuringTime()

        print("Timings for PE " + str(self.myPe()) + ":")
        table = [["", "em", "send", "recv", "total"]]
        lineNb = 1
        sep = {}
        row_totals = [0.0] * 4
        for C, charm_type in self.activeChares:
            sep[lineNb] = "------ " + str(C) + " as " + charm_type.__name__ + " ------"
            for em in self.classEntryMethods[charm_type.type_id][C]:
                if not em.profile: continue
                vals = em.times + [sum(em.times)]
                for i in range(len(row_totals)): row_totals[i] += vals[i]
                table.append([em.name] + [str(round(v, 3)) for v in vals])
                lineNb += 1
        sep[lineNb] = "-------------------------------------------------------"
        table.append([""] + [str(round(v, 3)) for v in row_totals])
        lineNb += 1
        sep[lineNb] = "-------------------------------------------------------"
        misc_overheads = [str(round(v, 3)) for v in self.lib.times]
        table.append(["reductions", ' ', ' ', misc_overheads[0], misc_overheads[0]])
        table.append(["custom reductions",   ' ', ' ', misc_overheads[1], misc_overheads[1]])
        table.append(["migrating out",  ' ', ' ', misc_overheads[2], misc_overheads[2]])
        lineNb += 3
        sep[lineNb] = "-------------------------------------------------------"
        row_totals[2] += sum(self.lib.times)
        row_totals[3] += sum(self.lib.times)
        table.append([""] + [str(round(v, 3)) for v in row_totals])
        lineNb += 1
        self.__printTable__(table, sep)
        for i in (0, 1):
            if i == 0:
                print("\nMessages sent: " + str(len(self.msg_send_sizes)))
                msgLens = self.msg_send_sizes
            else:
                print("\nMessages received: " + str(len(self.msg_recv_sizes)))
                msgLens = self.msg_recv_sizes
            if len(msgLens) == 0: msgLens = [0.0]
            msgSizeStats = [min(msgLens), sum(msgLens) / float(len(msgLens)), max(msgLens)]
            print("Message size in bytes (min / mean / max): " + str([str(v) for v in msgSizeStats]))
            print("Total bytes = " + str(round(sum(msgLens) / 1024.0 / 1024.0, 3)) + " MB")

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
            raise Charm4PyError("Charm++ version >= " + req_str + " required. " +
                               "Existing version is " + cur_str)

    def getTopoTreeEdges(self, pe, root_pe, pes=None, bfactor=4):
        """ Returns (parent, children) of 'pe' in a tree spanning the given 'pes',
            or all PEs if 'pes' is None
            If 'pes' is specified, 'root_pe' must be in the first position of 'pes',
            and 'pe' must be a member of 'pes' """
        return self.lib.getTopoTreeEdges(pe, root_pe, pes, bfactor)

    # TODO take into account situations where myPe and numPes could change (shrink/expand?) and possibly SMP mode in future
    def myPe(self):
        return self._myPe

    def numPes(self):
        return self._numPes

    def exit(self, exitCode=0):
        self.lib.CkExit(exitCode)

    def abort(self, msg):
        self.lib.CkAbort(msg)

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

    def registerNewChareType(self, name, source):
        exec(source, charm.dynamic_register)
        chare_type = charm.dynamic_register[name]
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
    return CharmLib(charm, Options, libcharm_path)


def profile_send_function(func):
    def func_with_profiling(*args, **kwargs):
        em = charm.runningEntryMethod
        em.startMeasuringSendTime()
        ret = func(*args, **kwargs)
        em.stopMeasuringSendTime()
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

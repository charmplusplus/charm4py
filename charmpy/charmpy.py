#
# @author Juan Galvez (jjgalvez@illinois.edu)
#
# Charm++ Python module that allows writing Charm++ programs in Python.
# Accesses C/C++ Charm shared library for core runtime functionality.
#
import sys
if sys.version_info < (2, 7, 0):
  print("charmpy requires Python 2.7 or higher")
  exit(1)
import os
if sys.version_info < (3, 0, 0):
  import cPickle
else:
  import pickle as cPickle
import time
import zlib
import json
import importlib
import inspect
import types
import ckreduction
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
  AUTO_FLUSH_WHEN = True
  QUIET = False

# A Chare class defined by a user can be used in 3 ways: (1) as a Mainchare, (2) to form groups,
# (3) to form arrays. To achieve this, Charmpy can register with the Charm++ library up to 3
# different types for the given class (a Mainchare, a Group and an Array), and each type will
# register its own entry methods, even though the definition (body) of the entry methods in Python is the same.
MAINCHARE, GROUP, ARRAY = range(3)
CHARM_TYPES = (MAINCHARE, GROUP, ARRAY)

class EntryMethod(object):
  def __init__(self, C, name, charm_type_id, profile=False):
    self.C = C          # class to which method belongs to
    self.name = name    # entry method name
    self.isCtor = False # true if method is constructor
    self.epIdx = -1     # entry method index assigned by Charm
    self.profile = profile
    if profile: self.times = [0.0, 0.0, 0.0]    # (time inside entry method, py send overhead, py recv overhead)
    if sys.version_info[0] < 3:
      method = getattr(C, name).__func__
    else:
      method = getattr(C, name)
    # a user class' method can have up to len(CHARM_TYPES) EntryMethod objects associated with it
    if not hasattr(method, 'entry_method_obj'): method.entry_method_obj = [None] * len(CHARM_TYPES)
    method.entry_method_obj[charm_type_id] = self
  def addTimes(self, times):
    for i,t in enumerate(times): self.times[i] += t

class ReadOnlies(object): # for backwards-compatibility. TODO Remove eventually
  def __new__(cls):
    raise CharmPyError('This ReadOnlies API is deprecated. Please refer to documentation/examples for correct usage')

class __ReadOnlies(object): pass

## Constants to detect type of contributors for reduction. Order should match enum extContributorType ##
(CONTRIBUTOR_TYPE_ARRAY,
CONTRIBUTOR_TYPE_GROUP,
CONTRIBUTOR_TYPE_NODEGROUP) = range(3)

class CharmPyError(Exception):
  def __init__(self, msg):
    super(CharmPyError, self).__init__(msg)
    self.message = msg

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

# Acts as Charm++ runtime at the Python level, and is a wrapper for the Charm++ shared library
class Charm(object):

  def __init__(self):
    self._myPe = -1
    self._numPes = -1
    self.registered = {}      # class -> set of Charm types (Mainchare, Group, Array) for which this class is registered
    self.register_order = []  # list of classes in registration order (all processes must use same order)
    self.chares = {}
    self.groups = []          # group instances on this PE (indexed by group ID)
    self.arrays = {}          # aid -> idx -> array element instance with index idx on this PE
    self.entryMethods = {}    # ep_idx -> EntryMethod object
    self.classEntryMethods = [{} for t in CHARM_TYPES]  # charm_type_id -> class -> list of EntryMethod objects
    self.proxyClasses      = [{} for t in CHARM_TYPES]  # charm_type_id -> class -> proxy class
    self.sendTime = 0.0       # for profiling
    self.msg_send_sizes = []  # for profiling
    self.msg_recv_sizes = []  # for profiling
    self.activeChares = set() # for profiling (active chares on this PE)
    self.opts = Options
    self.rebuildFuncs = [rebuildByteArray, rebuildArray, rebuildNumpyArray]
    cfgPath = None
    from os.path import expanduser
    cfgPath = expanduser("~") + '/charmpy.cfg'
    if not os.path.exists(cfgPath):
      cfgPath = os.path.dirname(__file__) + '/charmpy.cfg' # look in folder where charmpy.py is
      if not os.path.exists(cfgPath): cfgPath = None
    if cfgPath is None:
      print("charmpy.cfg not found")
      exit(1)
    cfg = json.load(open(cfgPath, 'r'))
    if cfg['libcharm_interface'] == 'ctypes':
      from charmlib_ctypes import CharmLib
    elif cfg['libcharm_interface'] == 'cffi':
      sys.path.append(os.path.dirname(__file__) + '/__cffi_objs__')
      from charmlib_cffi import CharmLib
    else:
      print("Unrecognized interface " + cfg['libcharm_interface'])
      exit(1)
    self.lib = CharmLib(self, Options, cfg.get('libcharm_path'))
    self.ReducerType = self.lib.ReducerType
    self.CkContributeToChare = self.lib.CkContributeToChare
    self.CkContributeToGroup = self.lib.CkContributeToGroup
    self.CkContributeToArray = self.lib.CkContributeToArray
    self.CkChareSend = self.lib.CkChareSend
    self.CkGroupSend = self.lib.CkGroupSend
    self.CkArraySend = self.lib.CkArraySend
    self.reducers = ckreduction.ReducerContainer(self)
    self.redMgr   = ckreduction.ReductionManager(self, self.reducers)
    self.mainchareRegistered = False
    self.entry_func = None  # entry point to Charm program. can be used in place of defining a Mainchare

  def handleGeneralError(self):
    import traceback
    errorType, error, stacktrace = sys.exc_info()
    print("----------------- Python Stack Traceback PE " + str(CkMyPe()) + " -----------------")
    traceback.print_tb(stacktrace, limit=None)
    CkAbort(errorType.__name__ + ": " + str(error))

  def recvReadOnly(self, msg):
    roData = cPickle.loads(msg)
    for name,obj in roData.items(): setattr(readonlies, name, obj)

  def buildMainchare(self, onPe, objPtr, ep, args):
    cid = (onPe, objPtr)  # chare ID (objPtr should be a Python int or long)
    if onPe != CkMyPe():  # TODO this check can probably be removed as I assume the runtime already does it
      raise CharmPyError("Received msg for chare not on this PE")
    if cid in self.chares: raise CharmPyError("Chare " + str(cid) + " already instantiated")
    em = self.entryMethods[ep]
    if not em.isCtor: raise CharmPyError("Specified mainchare entry method not constructor")
    self.currentChareId = cid
    obj = object.__new__(em.C)  # create object but don't call __init__
    super(em.C, obj).__init__() # call Mainchare class __init__ first
    if self.entry_func is not None: obj.main = self.entry_func
    obj.__init__(args)          # now call the user's __init__
    self.chares[cid] = obj
    if Options.PROFILING: self.activeChares.add((em.C, Mainchare))
    if CkMyPe() == 0: # broadcast readonlies
      roData = {}
      for attr in dir(readonlies):   # attr is string
        if attr.startswith("_") or attr.endswith("_"): continue
        roData[attr] = getattr(readonlies, attr)
      msg = cPickle.dumps(roData, Options.PICKLE_PROTOCOL)
      #print("Registering readonly data of size " + str(len(msg)))
      self.lib.CkRegisterReadonly(b"python_ro", b"python_ro", msg)

  def invokeEntryMethod(self, obj, em, msg, dcopy_start, t0):
    if Options.LOCAL_MSG_OPTIM and (msg[:7] == b"_local:"):
      args = obj.__removeLocal__(int(msg[7:]))
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
        if reducer.hasPostprocess: args[0] = reducer.postprocess(args[0])
    if Options.PROFILING:
      recv_overhead, initTime, self.sendTime = (time.time() - t0), time.time(), 0.0
    if Options.AUTO_FLUSH_WHEN: obj._checkWhen = set(obj._when_buffer.keys())
    getattr(obj, em.name)(*args)  # invoke entry method
    if Options.AUTO_FLUSH_WHEN and (len(obj._checkWhen) > 0): obj.__flushWhen__()
    if Options.PROFILING:
      em.addTimes([time.time() - initTime - self.sendTime, self.sendTime, recv_overhead])

  def recvChareMsg(self, chare_id, ep, msg, t0, dcopy_start):
    obj = self.chares[chare_id]
    self.invokeEntryMethod(obj, self.entryMethods[ep], msg, dcopy_start, t0)

  def recvGroupMsg(self, gid, ep, msg, t0, dcopy_start):
    if gid >= len(self.groups):
      while len(self.groups) <= gid: self.groups.append(None)
    em = self.entryMethods[ep]
    obj = self.groups[gid]
    if obj is None:
      #if CkMyPe() == 0: print("Group " + str(gid) + " not instantiated yet")
      if not em.isCtor: raise CharmPyError("Specified group entry method not constructor")
      obj = object.__new__(em.C)  # create object but don't call __init__
      Group.initMember(obj, gid)
      super(em.C, obj).__init__() # call Chare class __init__ first
      obj.__init__()              # now call the user's __init__
      self.groups[gid] = obj
      if Options.PROFILING: self.activeChares.add((em.C, Group))
    else:
      self.invokeEntryMethod(obj, em, msg, dcopy_start, t0)

  def recvArrayMsg(self, aid, index, ep, msg, t0, dcopy_start, migration=False, resumeFromSync=False):
    #print("Array msg received, aid=" + str(aid) + " arrIndex=" + str(index) + " ep=" + str(ep))
    if aid not in self.arrays: self.arrays[aid] = {}
    obj = self.arrays[aid].get(index)
    if obj is None: # not instantiated yet
      #if CkMyPe() == 0: print("Array element " + str(aid) + " index " + str(index) + " not instantiated yet")
      em = self.entryMethods[ep]
      if not em.isCtor: raise CharmPyError("Specified array entry method not constructor")
      if migration:
        obj = cPickle.loads(msg)
        obj.AtSync = types.MethodType(ArrayElem_AtSync, obj) # reset AtSync instance method
        obj._contributeInfo = self.lib.initContributeInfo(aid, index, CONTRIBUTOR_TYPE_ARRAY)
      else:
        obj = object.__new__(em.C)  # create object but don't call __init__
        Array.initMember(obj, aid, index)
        super(em.C, obj).__init__() # call Chare class __init__ first
        obj.__init__()              # now call the user's array element __init__
      self.arrays[aid][index] = obj
      if Options.PROFILING: self.activeChares.add((em.C, Array))
    else:
      if resumeFromSync:
        msg = cPickle.dumps(({},[]))
        ep = getattr(obj, 'resumeFromSync').entry_method_obj[ARRAY].epIdx
      self.invokeEntryMethod(obj, self.entryMethods[ep], msg, dcopy_start, t0)

  def packMsg(self, destObj, msgArgs):
    """Prepares a message for sending, given arguments to an entry method invocation.

      The message is the result of pickling `(header,args)` where header is a dict,
      and args the list of arguments. If direct-copy is enabled, arguments supporting
      the buffer interface will bypass pickling and their place in 'args' will be
      made empty. Instead, info to reconstruct these args at the destination will be
      put in the header, and this method will return a list of buffers for
      direct-copying of these args into a CkMessage at Charm side.

      If destination object exists on same PE as source, the args will be stored in
      '_local' buffer of destination obj (without copying), and the msg will be a
      small "tag" to retrieve the args from '_local' when the msg is delivered.

      Args:
          destObj: destination object if it exists on the same PE as source, otherwise None
          msgArgs: arguments to entry method

      Returns:
          2-tuple containing msg and list of direct-copy buffers

    """
    direct_copy_buffers = []
    dcopy_size = 0
    if destObj: # if dest obj is local
      localTag = destObj.__addLocal__(msgArgs)
      msg = ("_local:" + str(localTag)).encode()
    else:
      header = {}           # msg header
      direct_copy_hdr = []  # goes to header
      args = list(msgArgs)
      if self.lib.direct_copy_supported:
        for i,arg in enumerate(msgArgs):
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
    self.lastMsgLen = len(msg) + dcopy_size
    return (msg, direct_copy_buffers)

  # register class C in Charm
  def registerInCharm(self, C, charm_type, libRegisterFunc):
    charm_type_id = charm_type.type_id
    entryMethods = self.classEntryMethods[charm_type_id][C]
    #if CkMyPe() == 0: print("CharmPy:: Registering class " + C.__name__ + " in Charm with " + str(len(entryMethods)) + " entry methods " + str([e.name for e in entryMethods]))
    C.idx, startEpIdx = libRegisterFunc(C.__name__ + str(charm_type_id), len(entryMethods))
    #if CkMyPe() == 0: print("CharmPy:: Chare idx=" + str(C.idx) + " ctor Idx=" + str(startEpIdx))
    for i,em in enumerate(entryMethods):
      if i == 0: em.isCtor = True
      em.epIdx = startEpIdx + i
      self.entryMethods[em.epIdx] = em
    proxyClass = charm_type.__getProxyClass__(C)
    self.proxyClasses[charm_type_id][C] = proxyClass
    setattr(self, proxyClass.__name__, proxyClass) # save new class in my namespace
    globals()[proxyClass.__name__] = proxyClass    # save in module namespace (needed to pickle the proxy)

  # first callback from Charm++ shared library
  # this method registers classes with the shared library
  def registerMainModule(self):
    self._myPe   = self.lib.CkMyPe()
    self._numPes = self.lib.CkNumPes()

    # Charm++ library captures stdout/stderr. here we reset the streams with a buffering
    # policy that ensures that messages reach Charm++ in a timely fashion
    sys.stdout = os.fdopen(1,'wt',1)
    sys.stderr = os.fdopen(2,'wt',1)
    if self.myPe() != 0: self.lib.CkRegisterReadonly(b"python_null", b"python_null", None)

    if (self.myPe() == 0) and (not Options.QUIET):
      import platform
      out_msg = ("CharmPy> Running on Python " + str(platform.python_version()) +
                " (" + str(platform.python_implementation()) + "). Using '" +
                self.lib.name + "' interface to access Charm++")
      if self.lib.name != "cffi": out_msg += ", **WARNING**: cffi recommended for best performance"
      print(out_msg)

    for C in self.register_order:
      charm_types = self.registered[C]
      if Mainchare in charm_types: self.registerInCharm(C, Mainchare, self.lib.CkRegisterMainchare)
      if Group in charm_types: self.registerInCharm(C, Group, self.lib.CkRegisterGroup)
      if Array in charm_types: self.registerInCharm(C, Array, self.lib.CkRegisterArray)

  def registerAs(self, C, charm_type_id):
    if charm_type_id == MAINCHARE:
      if self.mainchareRegistered:
        raise CharmPyError("More than one entry point has been specified")
      self.mainchareRegistered = True
    charm_type = charm_type_id_to_class[charm_type_id]
    #print("CharmPy: Registering class " + C.__name__, "as", charm_type.__name__, "type_id=", charm_type_id, charm_type)
    self.classEntryMethods[charm_type_id][C] = [EntryMethod(C,m,charm_type_id) for m in charm_type.__baseEntryMethods__()]
    for m in dir(C):
      if not callable(getattr(C,m)): continue
      if m.startswith("__") and m.endswith("__"): continue  # filter out non-user methods
      if m in ["AtSync", "flushWhen", "contribute", "gather"]: continue
      #print(m)
      self.classEntryMethods[charm_type_id][C].append(EntryMethod(C,m,charm_type_id,profile=Options.PROFILING))
    self.registered[C].add(charm_type)

  # called by user (from Python) to register their Charm++ classes with the CharmPy runtime
  # by default a class is registered to work with both Groups and Arrays
  def register(self, C, collections=(GROUP, ARRAY)):
    if C in self.registered: return # already registered
    if (not hasattr(C, 'mro')) or (Chare not in C.mro()):
      raise CharmPyError("Only subclasses of Chare can be registered")

    self.registered[C] = set()
    if Mainchare in C.mro():
      self.registerAs(C, MAINCHARE)
    else:
      for charm_type_id in collections: self.registerAs(C, charm_type_id)
    self.register_order.append(C)

  def start(self, classes=[], modules=[], entry=None):
    """
    Start Charm++ program.

    IMPORTANT: classes must be registered in the same order on all processes. In
    other words, the arguments to this method must have the same ordering on all
    processes.

    Args:
        classes: list of Charm classes to register with runtime
        modules: list of names of modules containing Charm classes (all of the Charm
                 classes defined in the module will be registered). method will
                 always search module '__main__' for Charm classes even if no
                 arguments are passed to this method.
    """
    if Options.PROFILING: self.contribute = profile_send_function(self.contribute)
    if "++quiet" in sys.argv: Options.QUIET = True
    for C in classes: self.register(C)
    M = list(modules)
    if '__main__' not in M: M.append('__main__')
    for module_name in M:
      if module_name not in sys.modules: importlib.import_module(module_name)
      for C_name,C in inspect.getmembers(sys.modules[module_name], inspect.isclass):
        if C.__module__ != __name__ and hasattr(C, 'mro'):
          if Chare in C.mro():
            self.register(C)
          elif Group in C.mro() or Array in C.mro():
            raise CharmPyError("Refer to new API to create Arrays and Groups")
    if entry is not None:
      self.entry_func = entry
      self.register(DefaultMainchare)
    if not self.mainchareRegistered:
      raise CharmPyError("Can't start program because no main entry point has been specified")
    if len(self.registered) == 0:
      raise CharmPyError("Can't start Charm program because no Charm classes registered")
    self.lib.start()

  def arrayElemLeave(self, aid, index, sizing):
    if sizing:
      obj = self.arrays[aid][index]
      del obj._contributeInfo  # don't want to pickle this
      del obj.AtSync # method exists only in instance, not in class. remove to avoid error when unpickling
      obj.migMsg = cPickle.dumps(obj, Options.PICKLE_PROTOCOL)
      return obj.migMsg
    else:
      obj = self.arrays[aid].pop(index)
      if hasattr(obj,"migMsg"): msg = obj.migMsg
      else:
        msg = cPickle.dumps(obj, Options.PICKLE_PROTOCOL)
      return msg

  # Charm class level contribute function used by Array, Group for reductions
  def contribute(self, data, reducer, target, contributor):
    contribution = self.redMgr.prepare(data, reducer, contributor)
    contributeInfo = self.lib.getContributeInfo(target.ep, contribution, contributor)
    self.lastMsgLen = contributeInfo.dataSize
    target.__self__.ckContribute(contributeInfo)

  def printTable(self, table, sep):
    col_width = [max(len(x) for x in col) for col in zip(*table)]
    for j,line in enumerate(table):
      if j in sep: print(sep[j])
      print("| " + " | ".join("{:{}}".format(x, col_width[i]) for i, x in enumerate(line)) + " |")

  def printStats(self):
    if not Options.PROFILING:
      print("NOTE: called charm.printStats() but profiling is disabled")
      return
    print("Timings for PE " + str(self.myPe()) + ":")
    table = [["","em","send","recv","total"]]
    lineNb = 1
    sep = {}
    row_totals = [0.0] * 4
    for C,charm_type in self.activeChares:
      sep[lineNb] = "------ " + str(C) + " as " + charm_type.__name__ + " ------"
      for em in self.classEntryMethods[charm_type.type_id][C]:
        if not em.profile: continue
        vals = em.times + [sum(em.times)]
        for i in range(len(row_totals)): row_totals[i] += vals[i]
        table.append( [em.name] + [str(round(v,3)) for v in vals] )
        lineNb += 1
    sep[lineNb] = "-------------------------------------------------------"
    table.append([""] + [str(round(v,3)) for v in row_totals])
    lineNb += 1
    sep[lineNb] = "-------------------------------------------------------"
    misc_overheads = [str(round(v,3)) for v in self.lib.times]
    table.append(["reductions", ' ', ' ', misc_overheads[0], misc_overheads[0]])
    table.append(["custom reductions",   ' ', ' ', misc_overheads[1], misc_overheads[1]])
    table.append(["migrating out",  ' ', ' ', misc_overheads[2], misc_overheads[2]])
    lineNb += 3
    sep[lineNb] = "-------------------------------------------------------"
    row_totals[2] += sum(self.lib.times)
    row_totals[3] += sum(self.lib.times)
    table.append([""] + [str(round(v,3)) for v in row_totals])
    lineNb += 1
    self.printTable(table, sep)
    for i in (0,1):
      if i == 0:
        print("\nMessages sent: " + str(len(self.msg_send_sizes)))
        msgLens = self.msg_send_sizes
      else:
        print("\nMessages received: " + str(len(self.msg_recv_sizes)))
        msgLens = self.msg_recv_sizes
      if len(msgLens) == 0: msgLens = [0.0]
      msgSizeStats = [min(msgLens), sum(msgLens) / float(len(msgLens)), max(msgLens)]
      print("Message size in bytes (min / mean / max): " + str([str(v) for v in msgSizeStats]))
      print("Total bytes = " + str(round(sum(msgLens) / 1024.0 / 1024.0,3)) + " MB")

  def lib_version_check(self, commit_id_str):
    req_version = tuple([int(n) for n in open(os.path.dirname(__file__) + '/libcharm_version', 'r').read().split('.')])
    version = [int(n) for n in commit_id_str.split('-')[0][1:].split('.')]
    version = tuple(version + [int(commit_id_str.split('-')[1])])
    if version < req_version:
      req_str = '.'.join([str(n) for n in req_version])
      cur_str = '.'.join([str(n) for n in version])
      raise CharmPyError("Charm++ version >= " + req_str + " required. " +
                         "Existing version is " + cur_str)

  # TODO take into account situations where myPe and numPes could change (shrink/expand?) and possibly SMP mode in future
  def myPe(self): return self._myPe
  def numPes(self): return self._numPes
  def exit(self): self.lib.CkExit()
  def abort(self, msg): self.lib.CkAbort(msg)

charm    = Charm()
readonlies = __ReadOnlies()
Reducer  = charm.reducers  # put reference to reducers in module scope
CkMyPe   = charm.myPe
CkNumPes = charm.numPes
CkExit   = charm.exit
CkAbort  = charm.abort

def profile_send_function(func):
  def func_with_profiling(*args, **kwargs):
    sendInitTime = time.time()
    func(*args, **kwargs)
    charm.msg_send_sizes.append(charm.lastMsgLen)
    charm.sendTime += (time.time() - sendInitTime)
  if hasattr(func, 'ep'): func_with_profiling.ep = func.ep
  return func_with_profiling

# This decorator makes a wrapper for the chosen entry method 'func'.
# It is used so that the entry method is invoked only if the chare's member
# 'attrib_name' is equal to the first argument of the entry method.
# Entry method is guaranteed to be invoked (for any message order) as long as there
# are messages satisfying the condition if AUTO_FLUSH_WHEN = True. Otherwise user
# must call chare.flushWhen() when a chare modifies its condition attribute
def when(attrib_name):
  def _when(func):
    def entryMethod(self, *args, **kwargs):
      tag = args[0]
      if tag == getattr(self, attrib_name):
        func(self, *args) # expected msg, invoke entry method
        if Options.AUTO_FLUSH_WHEN and (tag == getattr(self, attrib_name)): self._checkWhen.discard(func.__name__)
      else:
        msgs = self._when_buffer.setdefault(func.__name__, {})
        msgs.setdefault(tag, []).append(args) # store, don't expect msg now
        self._checkWhen = set() # no entry method ran, so no need to check when buffers
    entryMethod.when_attrib_name = attrib_name
    entryMethod.func = func
    return entryMethod
  return _when

# ---------------------- Chare -----------------------

class Chare(object):
  def __init__(self):
    if hasattr(self, '_chare_initialized'): return
    # messages to this chare from chares in the same PE are stored here without copying
    # or pickling. _local is a fixed size array that implements a mem pool, where msgs
    # can be in non-consecutive positions, and the indexes of free slots are stored
    # as a linked list inside _local, with _local_free_head being the index of the
    # first free slot, _local[_local_free_head] is the index of next free slot and so on
    self._local = [i for i in range(1, Options.LOCAL_MSG_BUF_SIZE+1)]
    self._local[-1] = None
    self._local_free_head = 0
    self._when_buffer = {}
    self._chare_initialized = True

  def __addLocal__(self, msg):
    if self._local_free_head is None: raise CharmPyError("Local msg buffer full. Increase LOCAL_MSG_BUF_SIZE")
    h = self._local_free_head
    self._local_free_head = self._local[self._local_free_head]
    self._local[h] = msg
    return h

  def __removeLocal__(self, tag):
    msg = self._local[tag]
    self._local[tag] = self._local_free_head
    self._local_free_head = tag
    return msg

  def flushWhen(self):
    if len(self._when_buffer) > 0:
      self._checkWhen = set(self._when_buffer.keys())
      self.__flushWhen__()
      self._checkWhen = set()

  def __flushWhen__(self):
    for method_name in self._checkWhen:
      msgs = self._when_buffer[method_name]
      method = getattr(self, method_name)
      while True:
        attrib = getattr(self, method.when_attrib_name)
        msgs_now = msgs.pop(attrib, []) # check if expected msgs are stored
        if len(msgs_now) == 0:
          if len(msgs) == 0: self._when_buffer.pop(method_name)
          break
        for m in msgs_now: method.func(self, *m)

  def contribute(self, data, reducer_type, target):
    charm.contribute(data, reducer_type, target, self)

  def gather(self, data, target):
    charm.contribute(data, Reducer.gather, target, self)

  def AtSync(self): pass  # chares which are members of an Array will have this replaced with ArrayElem_AtSync

  def migrate(self, toPe):
    # print("[charmpy] Calling migrate, aid: ", self.thisProxy.aid, "ndims",
              # self.thisProxy.ndims, "index: ", self.thisIndex, "toPe", toPe)
    charm.lib.CkMigrate(self.thisProxy.aid, self.thisIndex, toPe)
# ----------------- Mainchare and Proxy --------------

def mainchare_proxy_ctor(proxy, cid):
  proxy.cid = cid

def mainchare_proxy_method_gen(ep): # decorator, generates proxy entry methods
  def proxy_entry_method(*args, **kwargs):
    me = args[0] # proxy
    destObj = None
    if Options.LOCAL_MSG_OPTIM: destObj = charm.chares.get(me.cid)
    msg = charm.packMsg(destObj, args[1:])
    charm.CkChareSend(me.cid, ep, msg)
  proxy_entry_method.ep = ep
  return proxy_entry_method

def mainchare_proxy_contribute(proxy, contributeInfo):
  charm.CkContributeToChare(contributeInfo, proxy.cid)

class Mainchare(Chare):

  type_id = MAINCHARE

  def __init__(self):
    if hasattr(self, '_chare_initialized'): return
    super(Mainchare,self).__init__()
    self._cid = charm.currentChareId
    self.thisProxy = charm.proxyClasses[MAINCHARE][self.__class__](self._cid)

  @classmethod
  def __baseEntryMethods__(cls): return ["__init__"]

  @classmethod
  def __getProxyClass__(C, cls):
    #print("Creating mainchare proxy class for class " + cls.__name__)
    M = dict()  # proxy methods
    for m in charm.classEntryMethods[MAINCHARE][cls]:
      if m.epIdx == -1: raise CharmPyError("Unregistered entry method")
      if Options.PROFILING: M[m.name] = profile_send_function(mainchare_proxy_method_gen(m.epIdx))
      else: M[m.name] = mainchare_proxy_method_gen(m.epIdx)
    M["__init__"] = mainchare_proxy_ctor
    M["ckContribute"] = mainchare_proxy_contribute # function called when target proxy is Mainchare
    return type(cls.__name__ + 'Proxy', (), M) # create and return proxy class

class DefaultMainchare(Mainchare):
  def __init__(self, args):
    self.main(args)

# ------------------ Group and Proxy  ----------------------

def group_proxy_ctor(proxy, gid):
  proxy.gid = gid
  proxy.elemIdx = -1 # next entry method call will be to elemIdx PE (broadcast if -1)

def group_proxy_elem(proxy, pe):  # group proxy [] overload method
  proxy_clone = proxy.__class__(proxy.gid)
  proxy_clone.elemIdx = pe
  return proxy_clone

def group_proxy_method_gen(ep): # decorator, generates proxy entry methods
  def proxy_entry_method(*args, **kwargs):
    me = args[0] # proxy
    destObj = None
    if Options.LOCAL_MSG_OPTIM and (me.elemIdx == charm.myPe()): destObj = charm.groups[me.gid]
    msg = charm.packMsg(destObj, args[1:])
    charm.CkGroupSend(me.gid, me.elemIdx, ep, msg)
  proxy_entry_method.ep = ep
  return proxy_entry_method

def group_ckNew_gen(C, epIdx):
  @classmethod    # make ckNew a class (not instance) method of proxy
  def group_ckNew(cls):
    #print("calling ckNew for class " + C.__name__ + " cIdx= " + str(C.idx))
    gid = charm.lib.CkCreateGroup(C.idx, epIdx)
    return charm.groups[gid].thisProxy # return instance of Proxy
  return group_ckNew

def group_proxy_contribute(proxy, contributeInfo):
  charm.CkContributeToGroup(contributeInfo, proxy.gid, proxy.elemIdx)

class Group(object):

  type_id = GROUP

  def __new__(cls, C):
    if (not hasattr(C, 'mro')) or (Chare not in C.mro()):
      raise CharmPyError("Only subclasses of Chare can be member of Group")
    return charm.proxyClasses[GROUP][C].ckNew()

  @classmethod
  def initMember(cls, obj, gid):
    obj.thisIndex = CkMyPe()
    obj.thisProxy = charm.proxyClasses[GROUP][obj.__class__](gid)
    obj._contributeInfo = charm.lib.initContributeInfo(gid, obj.thisIndex, CONTRIBUTOR_TYPE_GROUP)

  @classmethod
  def __baseEntryMethods__(cls): return ["__init__"]

  @classmethod
  def __getProxyClass__(C, cls):
    #print("Creating group proxy class for class " + cls.__name__)
    M = dict()  # proxy methods
    entryMethods = charm.classEntryMethods[GROUP][cls]
    for m in entryMethods:
      if m.epIdx == -1: raise CharmPyError("Unregistered entry method")
      if Options.PROFILING: M[m.name] = profile_send_function(group_proxy_method_gen(m.epIdx))
      else: M[m.name] = group_proxy_method_gen(m.epIdx)
    M["__init__"] = group_proxy_ctor
    M["__getitem__"] = group_proxy_elem
    M["ckNew"] = group_ckNew_gen(cls, entryMethods[0].epIdx)
    M["ckContribute"] = group_proxy_contribute # function called when target proxy is Group
    return type(cls.__name__ + 'GroupProxy', (), M) # create and return proxy class

# -------------------- Array and Proxy -----------------------

def array_proxy_ctor(proxy, aid, ndims):
  proxy.aid = aid
  proxy.ndims = ndims
  proxy.elemIdx = () # next entry method call will be to elemIdx array element (broadcast if empty tuple)

def array_proxy_elem(proxy, idx): # array proxy [] overload method
  proxy_clone = proxy.__class__(proxy.aid, proxy.ndims)
  if type(idx) == int: idx = (idx,)
  if len(idx) != proxy_clone.ndims:
    raise CharmPyError("Dimensions of index " + str(idx) + " don't match array dimensions")
  proxy_clone.elemIdx = tuple(idx)
  return proxy_clone

def array_proxy_method_gen(ep): # decorator, generates proxy entry methods
  def proxy_entry_method(*args, **kwargs):
    me = args[0]  # proxy
    destObj = None
    if Options.LOCAL_MSG_OPTIM and (len(me.elemIdx) > 0): destObj = charm.arrays[me.aid].get(me.elemIdx)
    msg = charm.packMsg(destObj, args[1:])
    charm.CkArraySend(me.aid, me.elemIdx, ep, msg)
  proxy_entry_method.ep = ep
  return proxy_entry_method

# NOTE: From user side ckNew can be used as follows:
# arrProxy.ckNew((dim1, dim2,...)) or arrProxy.ckNew(dim1)
# arrProxy.ckNew(ndims=2) - create an empty array of 2 dimensions
def array_ckNew_gen(C, epIdx):
  @classmethod    # make ckNew a class (not instance) method of proxy
  def array_ckNew(cls, dims=None, ndims=-1):
    #if CkMyPe() == 0: print("calling array ckNew for class " + C.__name__ + " cIdx=" + str(C.idx))
    # FIXME?, for now, if dims contains all zeros, will assume no bounds given
    if type(dims) == int: dims = (dims,)

    if dims is None and ndims == -1:
      raise CharmPyError("Bounds and number of dimensions for array cannot be empty in ckNew")
    elif dims is not None and ndims != -1 and ndims != len(dims):
      raise CharmPyError("Number of bounds should match number of dimensions")
    elif dims is None and ndims != -1: # create an empty array
      dims = (0,)*ndims

    aid = charm.lib.CkCreateArray(C.idx, dims, epIdx)
    return cls(aid, len(dims)) # return instance of Proxy
  return array_ckNew

def array_ckInsert_gen(epIdx):
  def array_ckInsert(proxy, index, onPE=-1):
    if type(index) == int: index = (index,)
    assert len(index) == proxy.ndims, "Invalid index dimensions passed to ckInsert"
    charm.lib.CkInsert(proxy.aid, index, epIdx, onPE) #TODO: add constructor params
  return array_ckInsert

def array_proxy_contribute(proxy, contributeInfo):
  charm.CkContributeToArray(contributeInfo, proxy.aid, proxy.elemIdx)

def array_proxy_doneInserting(proxy):
  charm.lib.CkDoneInserting(proxy.aid)

def ArrayElem_AtSync(obj):
  # directly call CkArraySend to bypass Python code
  # the call is equivalent to obj.thisProxy[obj.thisIndex].AtSync()
  charm.CkArraySend(obj.thisProxy.aid, obj.thisIndex, obj.thisProxy.AtSync.ep, (b'',[]))

class Array(object):

  type_id = ARRAY

  def __new__(cls, C, dims=None, ndims=-1):
    if (not hasattr(C, 'mro')) or (Chare not in C.mro()):
      raise CharmPyError("Only subclasses of Chare can be member of Array")
    return charm.proxyClasses[ARRAY][C].ckNew(dims, ndims)

  @classmethod
  def initMember(cls, obj, aid, index):
    obj.thisIndex = index
    obj.thisProxy = charm.proxyClasses[ARRAY][obj.__class__](aid, len(obj.thisIndex))
    # NOTE currently only used at Python level. proxy object in charm runtime currently has this set to true
    obj.usesAtSync = False
    obj._contributeInfo = charm.lib.initContributeInfo(aid, obj.thisIndex, CONTRIBUTOR_TYPE_ARRAY)
    obj.AtSync = types.MethodType(ArrayElem_AtSync, obj)

  @classmethod
  def __baseEntryMethods__(cls):
    # 2nd __init__ used to register migration constructor
    return ["__init__", "__init__", "AtSync"]

  @classmethod
  def __getProxyClass__(C, cls):
    #print("Creating array proxy class for class " + cls.__name__)
    M = dict()  # proxy methods
    entryMethods = charm.classEntryMethods[ARRAY][cls]
    for m in entryMethods:
      if m.epIdx == -1: raise CharmPyError("Unregistered entry method")
      if Options.PROFILING: M[m.name] = profile_send_function(array_proxy_method_gen(m.epIdx))
      else: M[m.name] = array_proxy_method_gen(m.epIdx)
    M["__init__"] = array_proxy_ctor
    M["__getitem__"] = array_proxy_elem
    M["ckNew"] = array_ckNew_gen(cls, entryMethods[0].epIdx)
    M["ckInsert"] = array_ckInsert_gen(entryMethods[0].epIdx)
    M["ckContribute"] = array_proxy_contribute # function called when target proxy is Array
    M["ckDoneInserting"] = array_proxy_doneInserting
    return type(cls.__name__ + 'ArrayProxy', (), M) # create and return proxy class


charm_type_id_to_class = (Mainchare, Group, Array)   # needs to match order of MAINCHARE, GROUP, ... "enum"

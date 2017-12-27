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
import itertools
from charmlib_ctypes import CharmLib

class Options:
  PROFILING = False
  PICKLE_PROTOCOL = -1    # -1 is highest protocol number
  ZLIB_COMPRESSION = 0    # 0 to 9
  LOCAL_MSG_OPTIM = True
  LOCAL_MSG_BUF_SIZE = 50
  AUTO_FLUSH_WHEN = True
  QUIET = False

class EntryMethod(object):
  def __init__(self, C, name, profile=False):
    self.C = C          # class to which method belongs to
    self.name = name    # entry method name
    self.isCtor = False # true if method is constructor
    self.epIdx = -1     # entry method index assigned by Charm
    self.profile = profile
    if profile: self.times = [0.0, 0.0, 0.0]    # (time inside entry method, py send overhead, py recv overhead)
    if sys.version_info < (3, 0, 0): getattr(C, name).__func__.em = self
    else: getattr(C, name).em = self
  def addTimes(self, times):
    for i,t in enumerate(times): self.times[i] += t

class Singleton(object):
  _instance = None
  def __new__(cls, *args, **kwargs):
    if not isinstance(cls._instance, cls):
      cls._instance = object.__new__(cls, *args, **kwargs)
    return cls._instance

class ReadOnlies(Singleton): pass

## Constants to detect type of contributors for reduction. Order should match enum extContributorType ##
(CONTRIBUTOR_TYPE_ARRAY,
CONTRIBUTOR_TYPE_GROUP,
CONTRIBUTOR_TYPE_NODEGROUP) = range(3)

# global tuple containing Python basic types
PYTHON_BASIC_TYPES = (int, float, bool)

class CharmPyError(Exception):
  def __init__(self, msg):
    super(CharmPyError, self).__init__(msg)
    self.message = msg

# Acts as Charm++ runtime at the Python level, and is a wrapper for the Charm++ shared library
class Charm(Singleton):

  def __init__(self):
    self.mainchareTypes = []
    self.chares = {}
    self.groupTypes = []  # group classes registered in runtime system
    self.groups = []      # group instances on this PE (indexed by group ID)
    self.arrayTypes = []  # array classes registered in runtime system
    self.arrays = {}      # aid -> dict[idx] -> array element instance with index idx on this PE
    self.entryMethods = {}                # ep_idx -> EntryMethod object
    self.classEntryMethods = {}           # class name -> list of EntryMethod objects
    self.proxyClasses = {}                # class name -> proxy class
    self.proxyTimes = 0.0 # for profiling
    self.msgLens = [0]    # for profiling
    self.lib = CharmLib(self, Options)
    self.ReducerType = self.lib.ReducerType
    self.CkContributeToChare = self.lib.CkContributeToChare
    self.CkContributeToGroup = self.lib.CkContributeToGroup
    self.CkContributeToArray = self.lib.CkContributeToArray
    self.CkChareSend = self.lib.CkChareSend
    self.CkGroupSend = self.lib.CkGroupSend
    self.CkArraySend = self.lib.CkArraySend

  def handleGeneralError(self):
    import traceback
    errorType, error, stacktrace = sys.exc_info()
    print("----------------- Python Stack Traceback PE " + str(CkMyPe()) + " -----------------")
    traceback.print_tb(stacktrace, limit=None)
    CkAbort(errorType.__name__ + ": " + str(error))

  def recvReadOnly(self, msg):
    roData = cPickle.loads(msg)
    ro = ReadOnlies()
    for name,obj in roData.items(): setattr(ro, name, obj)

  def buildMainchare(self, onPe, objPtr, ep, args):
    cid = (onPe, objPtr)  # chare ID (objPtr should be a Python int or long)
    if onPe != CkMyPe():  # TODO this check can probably be removed as I assume the runtime already does it
      raise CharmPyError("Received msg for chare not on this PE")
    if cid in self.chares: raise CharmPyError("Chare " + str(cid) + " already instantiated")
    em = self.entryMethods[ep]
    if not em.isCtor: raise CharmPyError("Specified mainchare entry method not constructor")
    self.currentChareId = cid
    self.chares[cid] = em.C(args) # call mainchare constructor
    if CkMyPe() == 0: # broadcast readonlies
      ro = ReadOnlies()
      roData = {}
      for attr in dir(ro):   # attr is string
        if attr.startswith("_") or attr.endswith("_"): continue
        roData[attr] = getattr(ro, attr)
      msg = cPickle.dumps(roData, Options.PICKLE_PROTOCOL)
      #print("Registering readonly data of size " + str(len(msg)))
      self.lib.CkRegisterReadonly(b"python_ro", b"python_ro", msg)

  def invokeEntryMethod(self, obj, em, msg, t0, compression):
    if Options.LOCAL_MSG_OPTIM and msg.startswith(b"_local"):
      args = obj.__removeLocal__(int(msg.split(b":")[1]))
    else:
      if compression: msg = zlib.decompress(msg)
      header,args = cPickle.loads(msg)
      if b"custom_reducer" in header and header[b"custom_reducer"] == "gather":
        args[0] = [tup[1] for tup in args[0]]
    if Options.PROFILING:
      recv_overhead, initTime, self.proxyTimes = (time.time() - t0), time.time(), 0.0
    if Options.AUTO_FLUSH_WHEN: obj._checkWhen = set(obj._when_buffer.keys())
    getattr(obj, em.name)(*args)  # invoke entry method
    if Options.AUTO_FLUSH_WHEN and (len(obj._checkWhen) > 0): obj.__flushWhen__()
    if Options.PROFILING:
      em.addTimes([time.time() - initTime - self.proxyTimes, self.proxyTimes, recv_overhead])

  def recvChareMsg(self, onPe, objPtr, ep, msg, t0):
    cid = (onPe, objPtr)  # chare ID
    obj = self.chares.get(cid)
    if obj is None: raise CharmPyError("Chare with id " + str(cid) + " not found")
    self.invokeEntryMethod(obj, self.entryMethods[ep], msg, t0, Options.ZLIB_COMPRESSION > 0)

  def recvGroupMsg(self, gid, ep, msg, t0):
    if gid >= len(self.groups):
      while len(self.groups) <= gid: self.groups.append(None)
    em = self.entryMethods[ep]
    obj = self.groups[gid]
    if obj is None:
      #if CkMyPe() == 0: print("Group " + str(gid) + " not instantiated yet")
      if not em.isCtor: raise CharmPyError("Specified group entry method not constructor")
      self.currentGroupID = gid
      self.groups[gid] = em.C()
    else:
      self.invokeEntryMethod(obj, em, msg, t0, False)

  def recvArrayMsg(self, aid, index, ep, msg, t0, migration=False, resumeFromSync=False):
    #print("Array msg received, aid=" + str(aid) + " arrIndex=" + str(index) + " ep=" + str(ep))
    if aid not in self.arrays: self.arrays[aid] = {}
    obj = self.arrays[aid].get(index)
    if obj is None: # not instantiated yet
      #if CkMyPe() == 0: print("Array element " + str(aid) + " index " + str(index) + " not instantiated yet")
      em = self.entryMethods[ep]
      if not em.isCtor: raise CharmPyError("Specified array entry method not constructor")
      self.currentArrayID = aid
      self.currentArrayElemIndex = index
      if migration:
        if Options.ZLIB_COMPRESSION > 0: msg = zlib.decompress(msg)
        obj = cPickle.loads(msg)
      else:
        obj = em.C()
      self.arrays[aid][index] = obj
    else:
      if resumeFromSync:
        msg = cPickle.dumps(({},[]))
        ep = getattr(obj, 'resumeFromSync').em.epIdx
      self.invokeEntryMethod(obj, self.entryMethods[ep], msg, t0, Options.ZLIB_COMPRESSION > 0)

  # packs arguments for remote method call (msgArgs) into a msg (bytearray) and
  # returns the msg. In general, it returns the pickled arguments, but if the
  # destination object exists in this PE, it will store the args in '_local'
  # buffer of destination obj (without copying) and return a small msg instead
  # with information to retrieve the args when the scheduler delivers it
  def packMsg(self, destObj, compress, msgArgs):
    if destObj: # if dest obj is local
      localTag = destObj.__addLocal__(msgArgs)
      msg = ("_local:" + str(localTag)).encode()
    else:
      msg = ({},msgArgs)  # first element is msg header
      msg = cPickle.dumps(msg, Options.PICKLE_PROTOCOL)
      if compress: msg = zlib.compress(msg, Options.ZLIB_COMPRESSION)
    self.lastMsgLen = len(msg)
    return msg

  # register class C in Charm
  def registerInCharm(self, C, libRegisterFunc):
    entryMethods = self.classEntryMethods[C.__name__]
    #if CkMyPe() == 0: print("CharmPy:: Registering class " + C.__name__ + " in Charm with " + str(len(entryMethods)) + " entry methods " + str([e.name for e in entryMethods]))
    C.idx, startEpIdx = libRegisterFunc(C.__name__, len(entryMethods))
    #if CkMyPe() == 0: print("CharmPy:: Chare idx=" + str(C.idx) + " ctor Idx=" + str(startEpIdx))
    for i,em in enumerate(entryMethods):
      if i == 0: em.isCtor = True
      em.epIdx = startEpIdx + i
      self.entryMethods[em.epIdx] = em
    proxyClass = C.__getProxyClass__()
    self.proxyClasses[C.__name__] = proxyClass
    setattr(self, proxyClass.__name__, proxyClass) # save new class in my namespace
    globals()[proxyClass.__name__] = proxyClass    # save in module namespace (needed to pickle the proxy)

  # first callback from Charm++ shared library
  # this method registers classes with the shared library
  def registerMainModule(self):
    # Charm++ library captures stdout/stderr. here we reset the streams with a buffering
    # policy that ensures that messages reach Charm++ in a timely fashion
    sys.stdout = os.fdopen(1,'wt',1)
    sys.stderr = os.fdopen(2,'wt',1)
    if CkMyPe() != 0: self.lib.CkRegisterReadonly(b"python_null", b"python_null", None)

    for C in self.mainchareTypes: self.registerInCharm(C, self.lib.CkRegisterMainchare)
    for C in self.groupTypes: self.registerInCharm(C, self.lib.CkRegisterGroup)
    for C in self.arrayTypes: self.registerInCharm(C, self.lib.CkRegisterArray)

  # called by user (from Python) to register their Charm++ classes with the CharmPy runtime
  def register(self, C):
    if C.__name__ in self.classEntryMethods: return   # already registered
    #print("CharmPy: Registering class " + C.__name__)
    self.classEntryMethods[C.__name__] = [EntryMethod(C,m) for m in C.__baseEntryMethods__()]
    for m in dir(C):
      if not callable(getattr(C,m)): continue
      if m.startswith("__") and m.endswith("__"): continue  # filter out non-user methods
      if m in ["AtSync"]: continue
      #print(m)
      self.classEntryMethods[C.__name__].append(EntryMethod(C,m,profile=Options.PROFILING))

    # TODO: or maybe, if useful somewhere else, just use a class attribute in base
    # class that tells me what it is
    types = [T.__name__ for T in C.mro()]
    if "Group" in types: self.groupTypes.append(C)
    elif "Mainchare" in types: self.mainchareTypes.append(C)
    elif "Array" in types: self.arrayTypes.append(C)

  # begin Charm++ program
  def start(self, classes=[]):
    if "++quiet" in sys.argv: Options.QUIET = True
    for C in classes: self.register(C)
    self.lib.start()

  def arrayElemLeave(self, aid, index, sizing):
    if sizing:
      obj = self.arrays[aid][index]
      obj.migMsg = cPickle.dumps(obj, Options.PICKLE_PROTOCOL)
      if Options.ZLIB_COMPRESSION > 0: obj.migMsg = zlib.compress(obj.migMsg, Options.ZLIB_COMPRESSION)
      return obj.migMsg
    else:
      obj = self.arrays[aid].pop(index)
      if hasattr(obj,"migMsg"): msg = obj.migMsg
      else:
        msg = cPickle.dumps(obj, Options.PICKLE_PROTOCOL)
        if Options.ZLIB_COMPRESSION > 0: msg = zlib.compress(msg, Options.ZLIB_COMPRESSION)
      return msg

  # Charm class level contribute function used by Array, Group for reductions
  def contribute(self, data, reducer_type, target, contributor):
    # nop reduction short circuit
    if reducer_type is None or reducer_type == Reducer.nop:
      reducer_type = Reducer.nop
      data = [None]
      contributeInfo = self.lib.getContributeInfo(target.ep, data, reducer_type, contributor)
      self.lastMsgLen = 0
      target.__self__.ckContribute(contributeInfo)
      return

    pyReducer = False

    if not callable(reducer_type):
      check_elems = data
      if not hasattr(data, '__len__'): check_elems = [data]
      for elem in check_elems:
        if type(elem) not in PYTHON_BASIC_TYPES:
          pyReducer = True
          break
    else:
      pyReducer = True

    # load reducer based on if it's Python or Charm
    if not pyReducer:
      if not hasattr(data, '__len__'): data = [data]
      reducer_type = reducer_type[1][type(data[0])] # choose Charm reducer based on first data element since it's homogenous
    else:
      if not callable(reducer_type):
        reducer_type = reducer_type[0] # we are using in-built Python reducers

      # TODO: Can remove following if block by moving the pre-processing and post-processing
      # to custom reducers. Custom reducers can be defined as objects which have the functions
      # `applyReducer`, `preProcessData`, `postProcessData`
      if reducer_type.__name__ == "gather":
        # append array index to data for sorting in reducer
        rednMsg = ({b"custom_reducer": reducer_type.__name__}, [[(contributor[1], data)]])
      else:
        rednMsg = ({b"custom_reducer": reducer_type.__name__}, [data])
      rednMsgPickle = cPickle.dumps(rednMsg, Options.PICKLE_PROTOCOL)
      data = rednMsgPickle # data for custom reducers is a custom reduction msg
      reducer_type = self.ReducerType.external_py # inform Charm about using external Py reducer

    contributeInfo = self.lib.getContributeInfo(target.ep, data, reducer_type, contributor)
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
    print("Timings for PE " + str(CkMyPe()) + ":")
    total, pyoverhead = sum(self.lib.times), sum(self.lib.times)
    table = [["","em","send","recv"]]
    lineNb = 1
    sep = {}
    for C,entryMethods in self.classEntryMethods.items():
      sep[lineNb] = "---- " + str(C) + " ----"
      for em in entryMethods:
        if not em.profile: continue
        vals = em.times
        table.append( [em.name] + [str(round(v,3)) for v in vals] )
        total += sum(vals)
        pyoverhead += (vals[1] + vals[2])
        lineNb += 1
    self.printTable(table, sep)
    print("Total Python recorded time = " + str(total))
    print("Python non-entry method time = " + str(pyoverhead))
    if self.lib.times[1] > 0:
      print("Time in custom reductions = " + str(self.lib.times[1]))
    msgLens = self.msgLens[1:]  # first element is 0
    print("\nMessages sent: " + str(len(msgLens)))
    msgSizeStats = [min(msgLens), sum(msgLens) / float(len(msgLens)), max(msgLens)]
    print("Message size in bytes (min / mean / max): " + str([str(v) for v in msgSizeStats]))
    print("Total bytes sent = " + str(round(sum(msgLens) / 1024.0 / 1024.0,3)) + " MB")

charm    = Charm() # Charm is a singleton, ok to import this file multiple times
CkMyPe   = charm.lib.CkMyPe
CkNumPes = charm.lib.CkNumPes
CkExit   = charm.lib.CkExit
CkAbort  = charm.lib.CkAbort

def profile_proxy(func):
  def func_with_profiling(*args, **kwargs):
    proxyInitTime = time.time()
    func(*args, **kwargs)
    charm.msgLens.append(charm.lastMsgLen)
    charm.proxyTimes += (time.time() - proxyInitTime)
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

# ---------------------- CkReducer -----------------------

class CkReducer(Singleton):
  def __init__(self):
    self.nop = charm.ReducerType.nop
    self.max = (self._max, {int: charm.ReducerType.max_int, float: charm.ReducerType.max_double})
    self.sum = (self._sum, {int: charm.ReducerType.sum_int, float: charm.ReducerType.sum_double})

  # python versions of built-in reducers
  def _max(self, contribs):
    return max(contribs)

  def _sum(self, contribs):
    return sum(contribs)

  def gather(self, contribs):
    # contribs will be a list of list of tuples
    # first element of tuple is always array index of chare
    return sorted(itertools.chain(*contribs))

# global singular instance of Reducer
Reducer = CkReducer()
# add reference to Reducer in charm object
charm.Reducer = Reducer


# ---------------------- Chare -----------------------

class Chare(object):
  def __init__(self):
    # messages to this chare from chares in the same PE are stored here without copying
    # or pickling. _local is a fixed size array that implements a mem pool, where msgs
    # can be in non-consecutive positions, and the indexes of free slots are stored
    # as a linked list inside _local, with _local_free_head being the index of the
    # first free slot, _local[_local_free_head] is the index of next free slot and so on
    self._local = [i for i in range(1, Options.LOCAL_MSG_BUF_SIZE+1)]
    self._local[-1] = None
    self._local_free_head = 0
    self._when_buffer = {}

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

# ----------------- Mainchare and Proxy --------------

def mainchare_proxy_ctor(proxy, cid):
  proxy.cid = cid

def mainchare_proxy_method_gen(ep): # decorator, generates proxy entry methods
  def proxy_entry_method(*args, **kwargs):
    me = args[0] # proxy
    destObj = None
    if Options.LOCAL_MSG_OPTIM: destObj = charm.chares.get(me.cid)
    msg = charm.packMsg(destObj, Options.ZLIB_COMPRESSION > 0, args[1:])
    charm.CkChareSend(me.cid, ep, msg)
  proxy_entry_method.ep = ep
  return proxy_entry_method

def mainchare_proxy_contribute(proxy, contributeInfo):
  charm.CkContributeToChare(contributeInfo, proxy.cid)

class Mainchare(Chare):
  def __init__(self):
    super(Mainchare,self).__init__()
    self.cid = charm.currentChareId
    self.thisProxy = charm.proxyClasses[self.__class__.__name__](self.cid)

  @classmethod
  def __baseEntryMethods__(cls): return ["__init__"]

  @classmethod
  def __getProxyClass__(cls):
    #print("Creating proxy class for class " + cls.__name__)
    M = dict()  # proxy methods
    for m in charm.classEntryMethods[cls.__name__]:
      if m.epIdx == -1: raise CharmPyError("Unregistered entry method")
      if Options.PROFILING: M[m.name] = profile_proxy(mainchare_proxy_method_gen(m.epIdx))
      else: M[m.name] = mainchare_proxy_method_gen(m.epIdx)
    M["__init__"] = mainchare_proxy_ctor
    M["ckContribute"] = mainchare_proxy_contribute # function called when target proxy is Mainchare
    return type(cls.__name__ + 'Proxy', (), M) # create and return proxy class

# ------------------ Group and Proxy  ----------------------

def group_proxy_ctor(proxy, gid):
  proxy.gid = gid
  proxy.elemIdx = -1 # next entry method call will be to elemIdx PE (broadcast if -1)

def group_proxy_elem(proxy, pe): # group proxy [] overload method
  proxy.elemIdx = pe
  return proxy

def group_proxy_method_gen(ep): # decorator, generates proxy entry methods
  def proxy_entry_method(*args, **kwargs):
    me = args[0] # proxy
    destObj = None
    if Options.LOCAL_MSG_OPTIM and (me.elemIdx == CkMyPe()): destObj = charm.groups[me.gid]
    msg = charm.packMsg(destObj, False, args[1:])
    charm.CkGroupSend(me.gid, me.elemIdx, ep, msg)
    me.elemIdx = -1
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
  proxy.elemIdx = -1

class Group(Chare):
  def __init__(self):
    super(Group,self).__init__()
    self.gid = charm.currentGroupID
    self.thisIndex = CkMyPe()
    self.thisProxy = charm.proxyClasses[self.__class__.__name__](self.gid)

  def contribute(self, data, reducer_type, target):
    contributor = (self.gid, self.thisIndex, CONTRIBUTOR_TYPE_GROUP)
    charm.contribute(data, reducer_type, target, contributor)

  def gather(self, data, target):
    contributor = (self.gid, self.thisIndex, CONTRIBUTOR_TYPE_GROUP)
    charm.contribute(data, Reducer.gather, target, contributor)

  @classmethod
  def __baseEntryMethods__(cls): return ["__init__"]

  @classmethod
  def __getProxyClass__(cls):
    #print("Creating proxy class for class " + cls.__name__)
    M = dict()  # proxy methods
    entryMethods = charm.classEntryMethods[cls.__name__]
    for m in entryMethods:
      if m.epIdx == -1: raise CharmPyError("Unregistered entry method")
      if Options.PROFILING: M[m.name] = profile_proxy(group_proxy_method_gen(m.epIdx))
      else: M[m.name] = group_proxy_method_gen(m.epIdx)
    M["__init__"] = group_proxy_ctor
    M["__getitem__"] = group_proxy_elem
    M["ckNew"] = group_ckNew_gen(cls, entryMethods[0].epIdx)
    M["ckContribute"] = group_proxy_contribute # function called when target proxy is Group
    return type(cls.__name__ + 'Proxy', (), M) # create and return proxy class

# -------------------- Array and Proxy -----------------------

def array_proxy_ctor(proxy, aid, ndims):
  proxy.aid = aid
  proxy.ndims = ndims
  proxy.elemIdx = () # next entry method call will be to elemIdx array element (broadcast if empty tuple)

def array_proxy_elem(proxy, idx): # array proxy [] overload method
  if type(idx) == int: idx = (idx,)
  if len(idx) != proxy.ndims:
    raise CharmPyError("Dimensions of index " + str(idx) + " don't match array dimensions")
  proxy.elemIdx = tuple(idx)
  return proxy

def array_proxy_method_gen(ep): # decorator, generates proxy entry methods
  def proxy_entry_method(*args, **kwargs):
    me = args[0]  # proxy
    destObj = None
    if Options.LOCAL_MSG_OPTIM and (len(me.elemIdx) > 0): destObj = charm.arrays[me.aid].get(me.elemIdx)
    msg = charm.packMsg(destObj, Options.ZLIB_COMPRESSION > 0, args[1:])
    charm.CkArraySend(me.aid, me.elemIdx, ep, msg)
    me.elemIdx = ()
  proxy_entry_method.ep = ep
  return proxy_entry_method

def array_ckNew_gen(C, epIdx):
  @classmethod    # make ckNew a class (not instance) method of proxy
  def array_ckNew(cls, dims):
    #if CkMyPe() == 0: print("calling array ckNew for class " + C.__name__ + " cIdx=" + str(C.idx))
    # FIXME?, for now, if dims contains all zeros, will assume no bounds given
    if type(dims) == int: dims = (dims,)
    aid = charm.lib.CkCreateArray(C.idx, dims, epIdx)
    return cls(aid, len(dims)) # return instance of Proxy
  return array_ckNew

def array_proxy_contribute(proxy, contributeInfo):
  charm.CkContributeToArray(contributeInfo, proxy.aid, proxy.elemIdx)
  proxy.elemIdx = ()

class Array(Chare):
  def __init__(self):
    super(Array,self).__init__()
    self.aid = charm.currentArrayID
    self.thisIndex = charm.currentArrayElemIndex
    self.thisProxy = charm.proxyClasses[self.__class__.__name__](self.aid, len(self.thisIndex))
    # NOTE currently only used at Python level. proxy object in charm runtime currently has this set to true
    self.usesAtSync = False
    if Options.PROFILING: self.contribute = profile_proxy(self.contribute)

  def AtSync(self):
    self.thisProxy[self.thisIndex].AtSync()

  def contribute(self, data, reducer_type, target):
    contributor = (self.aid, self.thisIndex, CONTRIBUTOR_TYPE_ARRAY)
    charm.contribute(data, reducer_type, target, contributor)

  def gather(self, data, target):
    contributor = (self.aid, self.thisIndex, CONTRIBUTOR_TYPE_ARRAY)
    charm.contribute(data, Reducer.gather, target, contributor)

  @classmethod
  def __baseEntryMethods__(cls):
    # 2nd __init__ used to register migration constructor
    return ["__init__", "__init__", "AtSync"]

  @classmethod
  def __getProxyClass__(cls):
    #print("Creating proxy class for class " + cls.__name__)
    M = dict()  # proxy methods
    entryMethods = charm.classEntryMethods[cls.__name__]
    for m in entryMethods:
      if m.epIdx == -1: raise CharmPyError("Unregistered entry method")
      if Options.PROFILING: M[m.name] = profile_proxy(array_proxy_method_gen(m.epIdx))
      else: M[m.name] = array_proxy_method_gen(m.epIdx)
    M["__init__"] = array_proxy_ctor
    M["__getitem__"] = array_proxy_elem
    M["ckNew"] = array_ckNew_gen(cls, entryMethods[0].epIdx)
    M["ckContribute"] = array_proxy_contribute # function called when target proxy is Array
    return type(cls.__name__ + 'Proxy', (), M) # create and return proxy class

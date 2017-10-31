#
# @author Juan Galvez (jjgalvez@illinois.edu)
#
# Charm++ Python module that allows writing Charm++ programs in Python.
# Accesses C/C++ Charm shared library for core runtime functionality.
#
import sys
import os
import ctypes
from ctypes import c_int, c_short, c_char, c_float, c_double, c_char_p, c_void_p, POINTER, CFUNCTYPE, Structure, sizeof
import cPickle
import inspect
import time
import zlib

PROFILING = True
PICKLE_PROTOCOL = -1    # -1 is highest protocol number
ZLIB_COMPRESSION = 0    # 0 to 9
LOCAL_MSG_OPTIM = True  # experimental; disable if there are issues, see [1] in TODO.txt


def arrayIndexToTuple(ndims, arrayIndex):
  if ndims <= 3: return tuple(ctypes.cast(arrayIndex, POINTER(c_int * ndims)).contents)
  else: return tuple(ctypes.cast(arrayIndex, POINTER(c_short * ndims)).contents)

class EntryMethod(object):
  def __init__(self, C, name, profile=False):
    self.C = C        # class to which method belongs to
    self.name = name  # entry method name
    self.isCtor = False # true if method is constructor
    self.epIdx = -1   # entry method index assigned by Charm
    self.profile = profile
    if profile: self.times = [0.0, 0.0, 0.0]    # (time inside entry method, py send overhead, py recv overhead)
  def addTimes(self, t1, t2, t3):
    self.times[0] += t1
    self.times[1] += t2
    self.times[2] += t3

class Singleton(type):
  _instances = {}
  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
      cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
    return cls._instances[cls]

class ReadOnlies(object):
  __metaclass__ = Singleton


# Import some useful structures defined on Charm side

### !!! The order of fields here should match the struct CkReductionTypesExt in ckreduction.h !!! ####
class ReducerTypes(Structure):
  _fields_ = [
    ("sum_int",     c_int),
    ("sum_float",   c_int),
    ("sum_double",  c_int),
    ("nop",         c_int)
  ]

class ContributeInfo(Structure):
  _fields_ = [
    ("cbEpIdx", c_int),               # index of entry point at reduction target
    ("data", c_void_p),               # data contributed for reduction
    ("numelems", c_int),              # number of elements in data
    ("dataSize", c_int),              # size of data in bytes
    ("redType", c_int),               # type of reduction (ReducerTypes)
    ("arrayId", c_int),               # array ID of the contributing chare array
    ("arrayIdx", POINTER(c_int)),     # index of the contributing chare array element
    ("ndims", c_int)                  # number of dimensions in index
  ]


# Acts as Charm++ runtime at the Python level, and is a wrapper for the Charm++ shared library
class Charm(object):
  __metaclass__ = Singleton

  def __init__(self):
    self.mainchareTypes = []
    self.chares = {}
    self.groupTypes = []  # group classes registered in runtime system
    self.groups = []      # group instances on this PE (indexed by group ID)
    self.arrayTypes = []  # array classes registered in runtime system
    self.arrays = {}      # aid -> dict[idx] -> array element instance with index idx on this PE
    self.entryMethods = {}                # ep_idx -> EntryMethod object
    self.classEntryMethods = {}           # class name -> list of EntryMethod objects
    self.classEntryMethods_byName = {}    # class name -> dict[ep name] -> EntryMethod object
    self.proxyClasses = {}                # class name -> proxy class
    self.initCharmLibrary()
    self.proxyTimes = 0.0 # for profiling
    self.msgLens = []
    self.localMsgs = {}
    self.localMsgCounter = 0
    self.ReducerType = ReducerTypes.in_dll(self.lib, "charm_reducers")
    self.ReducerTypeMap = {} # reducer type -> (ctypes type, python type), TODO consider changing to list
    self.buildReducerTypeMap()

  def recvReadOnly(self, msgSize, msg):
    msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
    roData = cPickle.loads(msg)
    ro = ReadOnlies()
    for name,obj in roData.iteritems(): setattr(ro, name, obj)

  def buildMainchare(self, onPe, objPtr, ep, argc, argv):
    cid = (onPe, objPtr)  # chare ID
    if onPe != CkMyPe():  # TODO this check can probably be removed as I assume the runtime already does it
      CkAbort("charmpy ERROR: received msg for chare not on this PE")
    if cid in self.chares: CkAbort("charmpy ERROR: chare " + str(cid) + " already instantiated")
    em = self.entryMethods[ep]
    if not em.isCtor: CkAbort("charmpy ERROR: specified mainchare entry method not constructor")
    self.currentChareId = cid
    self.chares[cid] = em.C([argv[i] for i in range(argc)]) # call mainchare constructor
    if CkMyPe() == 0:
      ro = ReadOnlies()
      roData = {}
      for attr in dir(ro):   # attr is string
        if attr.startswith("__") and attr.endswith("__"): continue
        roData[attr] = getattr(ro, attr)
      msg = cPickle.dumps(roData, PICKLE_PROTOCOL)
      #print "Registering readonly data of size", len(msg)
      self.lib.CkRegisterReadonlyExt("python_ro", "python_ro", len(msg), msg)

  def invokeEntryMethod(self, obj, em, msg, t0, local, compression):
    if local and msg.startswith("_local"):
      args = self.getLocalMsg(int(msg.split(":")[1]))
    else:
      if compression: msg = zlib.decompress(msg)
      args = cPickle.loads(msg)
    if PROFILING:
      recv_overhead, initTime, self.proxyTimes = (time.time() - t0), time.time(), 0.0
    getattr(obj, em.name)(*args)  # invoke entry method
    if PROFILING: em.addTimes( (time.time() - initTime - self.proxyTimes), self.proxyTimes, recv_overhead )

  def recvChareMsg(self, onPe, objPtr, ep, msgSize, msg):
    if PROFILING: t0 = time.time()
    if msgSize > 0: msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
    cid = (onPe, objPtr)  # chare ID
    obj = self.chares.get(cid)
    if obj is None: CkAbort("charmpy ERROR: chare with id " + str(cid) + " not found")
    self.invokeEntryMethod(obj, self.entryMethods[ep], msg, t0, LOCAL_MSG_OPTIM, ZLIB_COMPRESSION > 0)

  def recvGroupMsg(self, gid, ep, msgSize, msg):
    if PROFILING: t0 = time.time()
    if msgSize > 0: msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
    if gid >= len(self.groups):
      while len(self.groups) <= gid: self.groups.append(None)
    em = self.entryMethods[ep]
    obj = self.groups[gid]
    if obj is None:
      #if CkMyPe() == 0: print "Group", gid, "not instantiated yet"
      if not em.isCtor: CkAbort("charmpy ERROR: specified group entry method not constructor")
      self.currentGroupID = gid
      self.groups[gid] = em.C()
    else:
      self.invokeEntryMethod(obj, em, msg, t0, False, False)

  def recvArrayMsg(self, aid, ndims, arrayIndex, ep, msgSize, msg, migration=False, resumeFromSync=False):
    if PROFILING: t0 = time.time()
    arrIndex = arrayIndexToTuple(ndims, arrayIndex)
    if msgSize > 0: msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
    #print "Array msg received, aid=", aid, "ndims=", ndims, "arrIndex=", arrIndex, "ep=", ep
    if aid not in self.arrays: self.arrays[aid] = {}
    obj = self.arrays[aid].get(arrIndex)
    if obj is None: # not instantiated yet
      #if CkMyPe() == 0: print "Array element", aid, "index", arrIndex, "not instantiated yet"
      em = self.entryMethods[ep]
      if not em.isCtor: CkAbort("charmpy ERROR: specified array entry method not constructor")
      self.currentArrayID = aid
      self.currentArrayElemIndex = arrIndex
      if migration:
        if ZLIB_COMPRESSION > 0: msg = zlib.decompress(msg)
        if hasattr(em.C, "pack"):
          obj = em.C.__new__(em.C)    # TODO test this, we don't want to call the constructor
          obj.unpack(msg)
        else:
          obj = cPickle.loads(msg)
      else:
        obj = em.C()
      self.arrays[aid][arrIndex] = obj
    else:
      if resumeFromSync: return obj.resumeFromSync()
      self.invokeEntryMethod(obj, self.entryMethods[ep], msg, t0, LOCAL_MSG_OPTIM, ZLIB_COMPRESSION > 0)

  def addLocalMsg(self, data):
    self.localMsgCounter += 1
    self.localMsgs[self.localMsgCounter] = data
    return self.localMsgCounter

  def getLocalMsg(self, localMsgId):
    return self.localMsgs.pop(localMsgId)

  def sendToEntryMethod(self, sendFunc, local, compress, msgArgs, sendArgs):
    if PROFILING: proxyInitTime = time.time()
    if local:
      localMsgId = self.addLocalMsg(msgArgs)
      msg = "_local:" + str(localMsgId)
    else:
      msg = cPickle.dumps(msgArgs, PICKLE_PROTOCOL)
      if compress: msg = zlib.compress(msg, ZLIB_COMPRESSION)
    sendFunc(*(sendArgs + [msg, len(msg)]))
    if PROFILING:
      self.proxyTimes += (time.time() - proxyInitTime)
      self.msgLens.append(len(msg))

  # register class C in Charm
  def registerInCharm(self, C, libRegisterFunc):
    C.cname = ctypes.create_string_buffer(C.__name__)
    chareIdx, startEpIdx = c_int(0), c_int(0)
    entryMethods = self.classEntryMethods[C.__name__]
    #if CkMyPe() == 0: print "CharmPy:: Registering class", C.__name__, "in Charm with", len(entryMethods), "entry methods", [e.name for e in entryMethods]
    libRegisterFunc(C.cname, len(entryMethods), ctypes.byref(chareIdx), ctypes.byref(startEpIdx))
    chareIdx, startEpIdx = chareIdx.value, startEpIdx.value
    #if CkMyPe() == 0: print "CharmPy:: Chare idx=", chareIdx, "ctor Idx=", startEpIdx
    C.idx = chareIdx
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
    # Charm++ captures stdout, so we need to reopen the sys.stdout stream
    sys.stdout = os.fdopen(1,"w",0)
    if CkMyPe() != 0:
      self.lib.CkRegisterReadonlyExt("python_null", "python_null", 0, None)

    for C in self.mainchareTypes: self.registerInCharm(C, self.lib.CkRegisterMainChareExt)
    for C in self.groupTypes: self.registerInCharm(C, self.lib.CkRegisterGroupExt)
    for C in self.arrayTypes: self.registerInCharm(C, self.lib.CkRegisterArrayExt)

  # called by user (from Python) to register their Charm++ classes with the CharmPy runtime
  def register(self, C):
    if C.__name__ in self.classEntryMethods: return   # already registered
    #print "CharmPy: Registering class", C.__name__
    self.classEntryMethods[C.__name__] = [EntryMethod(C,m) for m in C.__baseEntryMethods__()]
    d = {}
    for em in self.classEntryMethods[C.__name__]:
      d[em.name] = em
    self.classEntryMethods_byName[C.__name__] = d

    for m,v in inspect.getmembers(C, predicate=inspect.ismethod):
      if m.startswith("__") and m.endswith("__"): continue  # filter out non-user methods
      if m in ["_pack", "pack", "AtSync"]: continue
      #print m
      em = EntryMethod(C,m,profile=True)
      self.classEntryMethods[C.__name__].append(em)
      self.classEntryMethods_byName[C.__name__][m] = em

    # TODO: or maybe, if useful somewhere else, just use a class attribute in base
    # class that tells me what it is
    types = [T.__name__ for T in C.mro()]
    if "Group" in types: self.groupTypes.append(C)
    elif "Mainchare" in types: self.mainchareTypes.append(C)
    elif "Array" in types: self.arrayTypes.append(C)

  # begin Charm++ program
  def start(self, classes=[]):
    for C in classes: self.register(C)

    self.argv_bufs = [ctypes.create_string_buffer(arg) for arg in sys.argv]
    LP_c_char = POINTER(c_char)
    argc = len(sys.argv)
    argv_p = (LP_c_char * (argc + 1))()
    for i,arg in enumerate(self.argv_bufs): argv_p[i] = arg
    self.lib.StartCharmExt.argtypes = (c_int, POINTER(LP_c_char)) # argc, argv
    self.lib.StartCharmExt(argc, argv_p)

  def arrayElemLeave(self, aid, ndims, arrayIndex, pdata, sizing):
    arrIndex = arrayIndexToTuple(ndims, arrayIndex)
    if sizing:
      obj = self.arrays[aid][arrIndex]
      obj.migMsg = obj._pack()
      if ZLIB_COMPRESSION > 0: obj.migMsg = zlib.compress(obj.migMsg, ZLIB_COMPRESSION)
      pdata = None
      return len(obj.migMsg)
    else:
      obj = self.arrays[aid].pop(arrIndex)
      if hasattr(obj,"migMsg"): msg = obj.migMsg
      else:
        msg = obj._pack()
        if ZLIB_COMPRESSION > 0: msg = zlib.compress(msg, ZLIB_COMPRESSION)
      data = ctypes.create_string_buffer(msg)
      #pdata[0] = ctypes.cast(data, c_void_p).value
      pdata = ctypes.cast(pdata, POINTER(POINTER(c_char)))
      pdata[0] = data
      # TODO could Python garbage collect the msg before charm++ copies it?
      return len(msg)

  def arrayElemJoin(self, aid, ndims, arrayIndex, ep, msg, msgSize):
    self.recvArrayMsg(aid, ndims, arrayIndex, ep, msgSize, msg, migration=True)

  def resumeFromSync(self, aid, ndims, arrayIndex):
    self.recvArrayMsg(aid, ndims, arrayIndex, -1, 0, None, resumeFromSync=True)

  def buildReducerTypeMap(self):
    # update this function as and when new reducer types are added to CharmPy
    self.ReducerTypeMap[self.ReducerType.sum_int] = (c_int, int)
    self.ReducerTypeMap[self.ReducerType.sum_float] = (c_float, float)
    self.ReducerTypeMap[self.ReducerType.sum_double] = (c_double, float)
    self.ReducerTypeMap[self.ReducerType.nop] = (None, None)

  # Notes: data is a void*, it must be type casted based on reducerType to Python type
  # returnBuffer must contain the cPickled form of type casted data, use char** to writeback
  def cpickleData(self, data, returnBuffer, dataSize, reducerType):
    dataTypeTuple = self.ReducerTypeMap[reducerType]
    numElems = 0
    pyData = None
    if reducerType == self.ReducerType.nop:
      pyData = []
    else:
      numElems = dataSize / sizeof(dataTypeTuple[0])
      pyData = ctypes.cast(data, POINTER(dataTypeTuple[0] * numElems)).contents
      pyData = [list(pyData)] # can use numpy arrays here if needed

    # if reduction result is one element, use base type
    if numElems == 1: pyData = pyData[0]

    #print "In charmpy. Data:", data, "dataSize:", dataSize, "numElems:", numElems, "reducerType:" reducerType

    pickledData = cPickle.dumps(pyData, PICKLE_PROTOCOL)
    pickledData = ctypes.create_string_buffer(pickledData)
    # cast returnBuffer to char** and make it point to pickledData
    returnBuffer = ctypes.cast(returnBuffer, POINTER(POINTER(c_char)))
    returnBuffer[0] = pickledData

    return len(pickledData)

  def initCharmLibrary(self):
    libcharm_env_var = os.environ.get("LIBCHARM_PATH")
    if libcharm_env_var != None:
      self.lib = ctypes.CDLL(libcharm_env_var)
    else:
      self.lib = ctypes.CDLL("libcharm.so")

    self.REGISTER_MAIN_MODULE_CB_TYPE = CFUNCTYPE(None)
    self.registerMainModuleCb = self.REGISTER_MAIN_MODULE_CB_TYPE(self.registerMainModule)
    self.lib.registerCkRegisterMainModuleCallback(self.registerMainModuleCb)

    #self.RECV_RO_CB_TYPE = CFUNCTYPE(None, c_int, c_char_p)
    self.RECV_RO_CB_TYPE = CFUNCTYPE(None, c_int, POINTER(c_char))
    self.recvReadOnlyCb = self.RECV_RO_CB_TYPE(self.recvReadOnly)
    self.lib.registerReadOnlyRecvExtCallback(self.recvReadOnlyCb)

    self.BUILD_MAINCHARE_CB_TYPE = CFUNCTYPE(None, c_int, c_void_p, c_int, c_int, POINTER(c_char_p))
    self.buildMainchareCb = self.BUILD_MAINCHARE_CB_TYPE(self.buildMainchare)
    self.lib.registerMainchareCtorExtCallback(self.buildMainchareCb)

    self.RECV_CHARE_CB_TYPE = CFUNCTYPE(None, c_int, c_void_p, c_int, c_int, POINTER(c_char))
    self.recvChareCb = self.RECV_CHARE_CB_TYPE(self.recvChareMsg)
    self.lib.registerChareMsgRecvExtCallback(self.recvChareCb)

    self.RECV_GROUP_CB_TYPE = CFUNCTYPE(None, c_int, c_int, c_int, POINTER(c_char))
    self.recvGroupCb = self.RECV_GROUP_CB_TYPE(self.recvGroupMsg)
    self.lib.registerGroupMsgRecvExtCallback(self.recvGroupCb)

    self.RECV_ARRAY_CB_TYPE = CFUNCTYPE(None, c_int, c_int, POINTER(c_int), c_int, c_int, POINTER(c_char))
    self.recvArrayCb = self.RECV_ARRAY_CB_TYPE(self.recvArrayMsg)
    self.lib.registerArrayMsgRecvExtCallback(self.recvArrayCb)

    self.ARRAY_ELEM_LEAVE_CB_TYPE = CFUNCTYPE(c_int, c_int, c_int, POINTER(c_int), POINTER(c_char_p), c_int)
    #self.ARRAY_ELEM_LEAVE_CB_TYPE = CFUNCTYPE(c_int, c_int, c_int, POINTER(c_int), POINTER(POINTER(c_char)), c_int)
    self.arrayLeaveCb = self.ARRAY_ELEM_LEAVE_CB_TYPE(self.arrayElemLeave)
    self.lib.registerArrayElemLeaveExtCallback(self.arrayLeaveCb)

    self.ARRAY_ELEM_JOIN_CB_TYPE = CFUNCTYPE(None, c_int, c_int, POINTER(c_int), c_int, POINTER(c_char), c_int)
    self.arrayJoinCb = self.ARRAY_ELEM_JOIN_CB_TYPE(self.arrayElemJoin)
    self.lib.registerArrayElemJoinExtCallback(self.arrayJoinCb)

    self.RESUME_FROM_SYNC_CB_TYPE = CFUNCTYPE(None, c_int, c_int, POINTER(c_int))
    self.resumeFromSyncCb = self.RESUME_FROM_SYNC_CB_TYPE(self.resumeFromSync)
    self.lib.registerArrayResumeFromSyncExtCallback(self.resumeFromSyncCb)

    # Args to cpickleData: data, return_buffer, data_size, reducer_type
    self.CPICKLE_DATA_CB_TYPE = CFUNCTYPE(c_int, c_void_p, POINTER(c_char_p), c_int, c_int)
    self.cpickleDataCb = self.CPICKLE_DATA_CB_TYPE(self.cpickleData)
    self.lib.registerCPickleDataExtCallback(self.cpickleDataCb)

    # the following line decreases performance, don't know why. seems to work fine without it
    #self.lib.CkArrayExtSend.argtypes = (c_int, POINTER(c_int), c_int, c_int, c_char_p, c_int)
    self.CkArrayExtSend = self.lib.CkArrayExtSend
    self.CkGroupExtSend = self.lib.CkGroupExtSend
    self.CkChareExtSend = self.lib.CkChareExtSend

  def printTable(self, table, sep):
      col_width = [max(len(x) for x in col) for col in zip(*table)]
      for j,line in enumerate(table):
        if j in sep: print sep[j]
        print "| " + " | ".join("{:{}}".format(x, col_width[i]) for i, x in enumerate(line)) + " |"

  def printStats(self):
    if not PROFILING:
      print "NOTE: profiling is disabled"
      return
    total, pyoverhead = 0.0, 0.0
    table = [["","em","send","recv"]]
    lineNb = 1
    sep = {}
    for C,entryMethods in self.classEntryMethods.iteritems():
      sep[lineNb] = "---- " + str(C) + " ----"
      for em in entryMethods:
        if not em.profile: continue
        vals = em.times
        table.append( [em.name] + [str(round(v,3)) for v in vals] )
        total += sum(vals)
        pyoverhead += (vals[1] + vals[2])
        lineNb += 1
    self.printTable(table, sep)
    print "Total Python recorded time=", total
    print "Python non-entry method time=", pyoverhead
    print "\nArray messages:", len(self.msgLens)
    print "Min msg size=", min(self.msgLens)
    print "Mean msg size=", sum(self.msgLens) / float(len(self.msgLens))
    print "Max msg size=", max(self.msgLens)
    print "Total msg len=", round(sum(self.msgLens) / 1024.0 / 1024.0,3), "MB"

charm = Charm() # Charm is a singleton, ok to import this file multiple times

def CkMyPe(): return charm.lib.CkMyPeHook()
def CkNumPes(): return charm.lib.CkNumPesHook()
def CkExit(): charm.lib.CkExit()
def CkAbort(msg): charm.lib.CmiAbort(msg)

# ----------------- Mainchare and Proxy --------------

def mainchare_proxy_ctor(proxy, cid):
  proxy.cid = cid

def mainchare_proxy_method_gen(ep): # decorator, generates proxy entry methods
  def proxy_entry_method(*args, **kwargs):
    proxy = args[0]
    sendArgs = [proxy.cid[0], proxy.cid[1], ep]
    local = LOCAL_MSG_OPTIM and (proxy.cid in charm.chares)
    charm.sendToEntryMethod(charm.CkChareExtSend, local, ZLIB_COMPRESSION > 0, args[1:], sendArgs)
  return proxy_entry_method

class Mainchare(object):
  def __init__(self):
    self.cid = charm.currentChareId
    self.thisProxy = charm.proxyClasses[self.__class__.__name__](self.cid)

  @classmethod
  def __baseEntryMethods__(cls): return ["__init__"]

  @classmethod
  def __getProxyClass__(cls):
    #print "Creating proxy class for class", cls.__name__
    M = dict()  # proxy methods
    for m in charm.classEntryMethods[cls.__name__]:
      if m.epIdx == -1: CkAbort("charmpy ERROR: unregistered entry method")
      M[m.name] = mainchare_proxy_method_gen(m.epIdx)
    M["__init__"] = mainchare_proxy_ctor
    return type(cls.__name__ + 'Proxy', (), M) # create and return proxy class

# ------------------ Group and Proxy  ----------------------

def group_proxy_ctor(proxy, gid):
  proxy.gid = gid
  # next entry method call will be to elemIdx PE (broadcast if -1)
  proxy.elemIdx = -1

def group_proxy_elem(proxy, pe): # group proxy [] overload method
  proxy.elemIdx = pe
  return proxy

def group_proxy_method_gen(ep): # decorator, generates proxy entry methods
  def proxy_entry_method(*args, **kwargs):
    proxy = args[0]
    sendArgs = [proxy.gid, proxy.elemIdx, ep]
    charm.sendToEntryMethod(charm.CkGroupExtSend, False, False, args[1:], sendArgs)
    proxy.elemIdx = -1
  return proxy_entry_method

def group_ckNew_gen(C, epIdx):
  @classmethod    # make ckNew a class (not instance) method of proxy
  def group_ckNew(cls):
    #print "calling ckNew for class", C.__name__, " cIdx=", C.idx
    gid = charm.lib.CkCreateGroupExt(C.idx, epIdx, None, 0)
    return charm.groups[gid].thisProxy
  return group_ckNew

class Group(object):
  def __init__(self):
    self.gid = charm.currentGroupID
    self.thisIndex = CkMyPe()
    self.thisProxy = charm.proxyClasses[self.__class__.__name__](self.gid)

  @classmethod
  def __baseEntryMethods__(cls): return ["__init__"]

  @classmethod
  def __getProxyClass__(cls):
    #print "Creating proxy class for class", cls.__name__
    M = dict()  # proxy methods
    entryMethods = charm.classEntryMethods[cls.__name__]
    for m in entryMethods:
      if m.epIdx == -1: CkAbort("charmpy ERROR: unregistered entry method")
      M[m.name] = group_proxy_method_gen(m.epIdx)
    M["__init__"] = group_proxy_ctor
    M["__getitem__"] = group_proxy_elem
    M["ckNew"] = group_ckNew_gen(cls, entryMethods[0].epIdx)
    return type(cls.__name__ + 'Proxy', (), M) # create and return proxy class

# -------------------- Array and Proxy -----------------------

def array_proxy_ctor(proxy, aid, ndims):
  proxy.aid = aid
  # next entry method call will be to elemIdx array element (broadcast if empty tuple)
  proxy.elemIdx = ()
  # C equivalent of elemIdx. Keep it as long as array dimensions strictly
  proxy.c_elemIdx = (ctypes.c_int * ndims)(-1)

def array_proxy_elem(proxy, idx): # array proxy [] overload method
  # Check that length of idx matches array dimensions
  if len(idx) != len(proxy.c_elemIdx):
    CkAbort("Non-matching dimensions of array index")

  proxy.elemIdx = tuple(idx)
  for i,v in enumerate(idx): proxy.c_elemIdx[i] = v
  return proxy

def array_proxy_method_gen(ep): # decorator, generates proxy entry methods
  def proxy_entry_method(*args, **kwargs):
    me = args[0]  # proxy
    local = LOCAL_MSG_OPTIM and (len(me.elemIdx) > 0) and (me.elemIdx in charm.arrays[me.aid])
    sendArgs = [me.aid, me.c_elemIdx, len(me.elemIdx), ep]
    charm.sendToEntryMethod(charm.CkArrayExtSend, local, ZLIB_COMPRESSION > 0, args[1:], sendArgs)
    me.elemIdx = ()
  return proxy_entry_method

def array_ckNew_gen(C, epIdx):
  @classmethod    # make ckNew a class (not instance) method of proxy
  def array_ckNew(cls, dims):
    #if CkMyPe() == 0: print "calling array ckNew for class", C.__name__, " cIdx=", C.idx
    # FIXME?, for now, if dims contains all zeros, will assume no bounds given
    ndims = len(dims)
    dimsArray = (c_int*ndims)(*dims)
    #if CkMyPe() == 0: print "ndims=", ndims, "dimsArray=", [dimsArray[i] for i in range(ndims)], "epIdx=", charm.classEntryMethods[C.__name__][0].epIdx
    aid = charm.lib.CkCreateArrayExt(C.idx, ndims, dimsArray, epIdx, None, 0)
    return cls(aid, ndims) # return instance of Proxy
  return array_ckNew

class Array(object):
  def __init__(self):
    self.aid = charm.currentArrayID
    self.thisIndex = charm.currentArrayElemIndex
    self.thisProxy = charm.proxyClasses[self.__class__.__name__](self.aid, len(self.thisIndex))
    # NOTE currently only used at Python level. proxy object in charm runtime currently has this set to true
    self.usesAtSync = False

  def AtSync(self):
    self.thisProxy[self.thisIndex].AtSync()

  def contribute(self, data, reducer_type, entry_method, proxy):
    target_class = entry_method.im_class.__name__
    target_ep_name = entry_method.__name__
    try:
      target_ep_object = charm.classEntryMethods_byName[target_class][target_ep_name]
    except KeyError:
      CkAbort("charmpy ERROR: reduction target not registered")
    target_ep_idx = target_ep_object.epIdx

    # determine length of data being contributed
    # TODO consider numpy or other array types too
    if type(data) != list: data = [data]
    numElems = len(data)

    # converting data to c_data for Charm side
    c_data = None
    c_data_size = 0
    if reducer_type != charm.ReducerType.nop:
      dataTypeTuple = charm.ReducerTypeMap[reducer_type]
      c_data = (dataTypeTuple[0]*numElems)(*data)
      c_data_size = numElems*sizeof(dataTypeTuple[0])

    #print data, reducer_type, target_class, target_ep_name, target_ep_idx, type(proxy).__name__, numElems, c_data
    c_arrayIdx = (c_int*len(self.thisIndex))(*self.thisIndex)

    contributeInfo = ContributeInfo(target_ep_idx, ctypes.cast(c_data, c_void_p), numElems, c_data_size,
                                    reducer_type, self.aid, c_arrayIdx, len(self.thisIndex))

    proxy_type = entry_method.im_class.__bases__[0].__name__
    if proxy_type == 'Mainchare': # array contributing to main chare
      charm.lib.CkArrayExtContributeChare(ctypes.byref(contributeInfo), proxy.cid[0], proxy.cid[1])
    elif proxy_type == 'Array': # array contributing to an array element/bcast
      charm.lib.CkArrayExtContributeArray(ctypes.byref(contributeInfo), proxy.aid, proxy.c_elemIdx, len(proxy.elemIdx))
      proxy.elemIdx = ()
    else: # using placeholder
      charm.lib.CkArrayExtContributeTmp(contributeInfo)


  @classmethod
  def __baseEntryMethods__(cls):
    # 2nd __init__ used to register migration constructor
    return ["__init__", "__init__", "AtSync"]

  @classmethod
  def __getProxyClass__(cls):
    #print "Creating proxy class for class", cls.__name__
    M = dict()  # proxy methods
    entryMethods = charm.classEntryMethods[cls.__name__]
    for m in entryMethods:
      if m.epIdx == -1: CkAbort("charmpy ERROR: unregistered entry method")
      M[m.name] = array_proxy_method_gen(m.epIdx)
    M["__init__"] = array_proxy_ctor
    M["__getitem__"] = array_proxy_elem
    M["ckNew"] = array_ckNew_gen(cls, entryMethods[0].epIdx)
    return type(cls.__name__ + 'Proxy', (), M) # create and return proxy class

  def _pack(self):
    if hasattr(self, "pack"): return self.pack()  # user-implemented pack method
    else: return cPickle.dumps(self, PICKLE_PROTOCOL) # pickle my whole self


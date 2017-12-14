import ctypes
from ctypes import c_int, c_short, c_char, c_float, c_double, c_char_p, c_void_p, POINTER, CFUNCTYPE, Structure, sizeof
import sys
import os
import time
if sys.version_info < (3, 0, 0):
  import cPickle
else:
  import pickle as cPickle

# Import some useful structures defined on Charm side

### !!! The order of fields here should match the struct CkReductionTypesExt in ckreduction.h !!! ####
class ReducerTypes(Structure):
  _fields_ = [
    ("sum_int",     c_int),
    ("sum_float",   c_int),
    ("sum_double",  c_int),
    ("nop",         c_int),
    ("max_int",     c_int),
    ("max_float",   c_int),
    ("max_double",  c_int),
    ("external_py", c_int)
  ]

class ContributeInfo(Structure):
  _fields_ = [
    ("cbEpIdx", c_int),               # index of entry point at reduction target
    ("data", c_void_p),               # data contributed for reduction
    ("numelems", c_int),              # number of elements in data
    ("dataSize", c_int),              # size of data in bytes
    ("redType", c_int),               # type of reduction (ReducerTypes)
    ("id", c_int),                    # ID of the contributing array/group
    ("idx", POINTER(c_int)),          # index of the contributing chare array/group element
    ("ndims", c_int),                 # number of dimensions in index
    ("contributorType", c_int)        # type of contributor
  ]

class CharmLib(object):

  def __init__(self, charm, opts):
    self.chareNames = []
    self.charm = charm
    self.opts = opts
    self.init()
    self.ReducerType = ReducerTypes.in_dll(self.lib, "charm_reducers")
    self.ReducerTypeMap = {} # reducer type -> ctypes type, TODO consider changing to list
    self.buildReducerTypeMap()

  def buildReducerTypeMap(self):
    # update this function as and when new reducer types are added to CharmPy
    self.ReducerTypeMap[self.ReducerType.sum_int] = c_int
    self.ReducerTypeMap[self.ReducerType.sum_float] = c_float
    self.ReducerTypeMap[self.ReducerType.sum_double] = c_double
    self.ReducerTypeMap[self.ReducerType.nop] = None
    self.ReducerTypeMap[self.ReducerType.max_int] = c_int
    self.ReducerTypeMap[self.ReducerType.max_float] = c_float
    self.ReducerTypeMap[self.ReducerType.max_double] = c_double
    self.ReducerTypeMap[self.ReducerType.external_py] = c_char

  def getContributeInfo(self, ep, data, reducer_type, contributor):
    if reducer_type is None: reducer_type = self.ReducerType.nop
    numElems = len(data)
    c_data = None
    c_data_size = 0
    if reducer_type != self.ReducerType.nop:
      dataType = self.ReducerTypeMap[reducer_type]
      c_data = (dataType*numElems)(*data)
      c_data_size = numElems*sizeof(dataType)

    elemId, index, elemType = contributor
    if type(index) == int: index = [index]
    ndims = len(index)
    c_elemIdx = (c_int*ndims)(*index)
    return ContributeInfo(ep, ctypes.cast(c_data, c_void_p), numElems, c_data_size,
                          reducer_type, elemId, c_elemIdx, ndims, elemType)

  def arrayIndexToTuple(self, ndims, arrayIndex):
    if ndims <= 3: return tuple(ctypes.cast(arrayIndex, POINTER(c_int * ndims)).contents)
    else: return tuple(ctypes.cast(arrayIndex, POINTER(c_short * ndims)).contents)

  def recvReadOnly(self, msgSize, msg):
    try:
      msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
      self.charm.recvReadOnly(msg)
    except:
      self.charm.handleGeneralError()

  def buildMainchare(self, onPe, objPtr, ep, argc, argv):
    try:
      self.charm.buildMainchare(onPe, objPtr, ep, [argv[i].decode() for i in range(argc)])
    except:
      self.charm.handleGeneralError()

  def recvChareMsg(self, onPe, objPtr, ep, msgSize, msg):
    try:
      t0 = None
      if self.opts.PROFILING: t0 = time.time()
      if msgSize > 0: msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
      self.charm.recvChareMsg(onPe, objPtr, ep, msg, t0)
    except:
      self.charm.handleGeneralError()

  def recvGroupMsg(self, gid, ep, msgSize, msg):
    try:
      t0 = None
      if self.opts.PROFILING: t0 = time.time()
      if msgSize > 0: msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
      self.charm.recvGroupMsg(gid, ep, msg, t0)
    except:
      self.charm.handleGeneralError()

  def recvArrayMsg(self, aid, ndims, arrayIndex, ep, msgSize, msg):
    try:
      t0 = None
      if self.opts.PROFILING: t0 = time.time()
      arrIndex = self.arrayIndexToTuple(ndims, arrayIndex)
      if msgSize > 0: msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
      self.charm.recvArrayMsg(aid, arrIndex, ep, msg, t0)
    except:
      self.charm.handleGeneralError()

  def CkChareSend(self, chare_id, ep, msg):
    self.lib.CkChareExtSend(chare_id[0], chare_id[1], ep, msg, len(msg))

  def CkGroupSend(self, group_id, index, ep, msg):
    self.lib.CkGroupExtSend(group_id, index, ep, msg, len(msg))

  def CkArraySend(self, array_id, index, ep, msg):
    ndims = len(index)
    c_elemIdx = (ctypes.c_int * ndims)(*index)  # TODO have buffer preallocated for this?
    self.lib.CkArrayExtSend(array_id, c_elemIdx, ndims, ep, msg, len(msg))

  def CkRegisterReadonly(self, n1, n2, msg):
    if msg is None: self.lib.CkRegisterReadonlyExt(n1, n2, 0, msg)
    else: self.lib.CkRegisterReadonlyExt(n1, n2, len(msg), msg)

  def CkRegisterMainchare(self, name, numEntryMethods):
    self.chareNames.append(ctypes.create_string_buffer(name.encode()))
    chareIdx, startEpIdx = c_int(0), c_int(0)
    self.lib.CkRegisterMainChareExt(self.chareNames[-1], numEntryMethods, ctypes.byref(chareIdx), ctypes.byref(startEpIdx))
    return int(chareIdx.value), int(startEpIdx.value)

  def CkRegisterGroup(self, name, numEntryMethods):
    self.chareNames.append(ctypes.create_string_buffer(name.encode()))
    chareIdx, startEpIdx = c_int(0), c_int(0)
    self.lib.CkRegisterGroupExt(self.chareNames[-1], numEntryMethods, ctypes.byref(chareIdx), ctypes.byref(startEpIdx))
    return int(chareIdx.value), int(startEpIdx.value)

  def CkRegisterArray(self, name, numEntryMethods):
    self.chareNames.append(ctypes.create_string_buffer(name.encode()))
    chareIdx, startEpIdx = c_int(0), c_int(0)
    self.lib.CkRegisterArrayExt(self.chareNames[-1], numEntryMethods, ctypes.byref(chareIdx), ctypes.byref(startEpIdx))
    return int(chareIdx.value), int(startEpIdx.value)

  def CkCreateGroup(self, chareIdx, epIdx):
    return self.lib.CkCreateGroupExt(chareIdx, epIdx, None, 0)

  def CkCreateArray(self, chareIdx, dims, epIdx):
    ndims = len(dims)
    dimsArray = (c_int*ndims)(*dims)
    return self.lib.CkCreateArrayExt(chareIdx, ndims, dimsArray, epIdx, None, 0)

  def start(self):
    self.argv_bufs = [ctypes.create_string_buffer(arg.encode()) for arg in sys.argv]
    LP_c_char = POINTER(c_char)
    argc = len(sys.argv)
    argv_p = (LP_c_char * (argc + 1))()
    for i,arg in enumerate(self.argv_bufs): argv_p[i] = arg
    self.lib.StartCharmExt.argtypes = (c_int, POINTER(LP_c_char)) # argc, argv
    self.lib.StartCharmExt(argc, argv_p)

  def arrayElemLeave(self, aid, ndims, arrayIndex, pdata, sizing):
    try:
      arrIndex = self.arrayIndexToTuple(ndims, arrayIndex)
      msg = self.charm.arrayElemLeave(aid, arrIndex, bool(sizing))
      if sizing:
        pdata = None
        return len(msg)
      else:
        data = ctypes.create_string_buffer(msg)
        #pdata[0] = ctypes.cast(data, c_void_p).value
        pdata = ctypes.cast(pdata, POINTER(POINTER(c_char)))
        pdata[0] = data
        # TODO could Python garbage collect the msg before charm++ copies it?
        return len(msg)
    except:
      self.charm.handleGeneralError()

  def arrayElemJoin(self, aid, ndims, arrayIndex, ep, msg, msgSize):
    try:
      t0 = None
      if self.opts.PROFILING: t0 = time.time()
      arrIndex = self.arrayIndexToTuple(ndims, arrayIndex)
      if msgSize > 0: msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
      self.charm.recvArrayMsg(aid, arrIndex, ep, msg, t0, migration=True)
    except:
      self.charm.handleGeneralError()

  def resumeFromSync(self, aid, ndims, arrayIndex):
    try:
      t0 = None
      if self.opts.PROFILING: t0 = time.time()
      arrIndex = self.arrayIndexToTuple(ndims, arrayIndex)
      self.charm.recvArrayMsg(aid, arrIndex, -1, 0, t0, resumeFromSync=True)
    except:
      self.charm.handleGeneralError()

  def CkContributeToChare(self, contributeInfo, cid):
    self.lib.CkExtContributeToChare(ctypes.byref(contributeInfo), cid[0], cid[1])

  def CkContributeToGroup(self, contributeInfo, gid, elemIdx):
    self.lib.CkExtContributeToGroup(ctypes.byref(contributeInfo), gid, elemIdx)

  def CkContributeToArray(self, contributeInfo, aid, index):
    ndims = len(index)
    c_elemIdx = (ctypes.c_int * ndims)(*index)
    self.lib.CkExtContributeToArray(ctypes.byref(contributeInfo), aid, c_elemIdx, ndims)

  # Notes: data is a void*, it must be type casted based on reducerType to Python type
  # returnBuffer must contain the cPickled form of type casted data, use char** to writeback
  def cpickleData(self, data, returnBuffer, dataSize, reducerType):
    try:
      dataType = self.ReducerTypeMap[reducerType]
      numElems = 0
      pyData = None
      if reducerType == self.ReducerType.nop:
        pyData = []
      else:
        numElems = dataSize // sizeof(dataType)
        pyData = ctypes.cast(data, POINTER(dataType * numElems)).contents
        pyData = [list(pyData)] # can use numpy arrays here if needed

      # if reduction result is one element, use base type
      if numElems == 1: pyData = pyData[0]

      #print("In charmpy. Data: " + str(data) + " dataSize: " + str(dataSize) + " numElems: " + str(numElems) + " reducerType: " + str(reducerType))

      msg = ({},pyData) # first element is msg header
      pickledData = cPickle.dumps(msg, self.opts.PICKLE_PROTOCOL)
      pickledData = ctypes.create_string_buffer(pickledData)
      # cast returnBuffer to char** and make it point to pickledData
      returnBuffer = ctypes.cast(returnBuffer, POINTER(POINTER(c_char)))
      returnBuffer[0] = pickledData

      return len(pickledData)
    except:
      self.charm.handleGeneralError()

  # callback function invoked by Charm++ for reducing contributions using a Python reducer (built-in or custom)
  def pyReduction(self, msgs, msgSizes, nMsgs, returnBuffer):
    contribs = []
    currentReducer = None
    for i in range(nMsgs):
      msg = msgs[i]
      if msgSizes[i] > 0:
        msg = ctypes.cast(msg, POINTER(c_char * msgSizes[i])).contents.raw
        header, args = cPickle.loads(msg)

        customReducer = header[b"custom_reducer"]

        if currentReducer is None: currentReducer = customReducer
        # check for correctness of msg
        assert customReducer == currentReducer

        contribs.append(args[0])

    reductionResult = getattr(self.charm.Reducer, currentReducer)(contribs)
    rednMsg = ({b"custom_reducer": currentReducer}, [reductionResult])
    rednMsgPickle = cPickle.dumps(rednMsg, self.opts.PICKLE_PROTOCOL)
    rednMsgPickle = ctypes.create_string_buffer(rednMsgPickle)

    # cast returnBuffer to char** and make it point to pickled reduction msg
    returnBuffer = ctypes.cast(returnBuffer, POINTER(POINTER(c_char)))
    returnBuffer[0] = rednMsgPickle

    return len(rednMsgPickle)

  # first callback from Charm++ shared library
  def registerMainModule(self):
    try:
      self.charm.registerMainModule()
    except:
      self.charm.handleGeneralError()

  def init(self):
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

    # Args to pyReduction: msgs, msgSizes, nMsgs, returnBuffer
    self.PY_REDUCTION_CB_TYPE = CFUNCTYPE(c_int, POINTER(c_void_p), POINTER(c_int), c_int, POINTER(c_char_p))
    self.pyReductionCb = self.PY_REDUCTION_CB_TYPE(self.pyReduction)
    self.lib.registerPyReductionExtCallback(self.pyReductionCb)

    # the following line decreases performance, don't know why. seems to work fine without it
    #self.lib.CkArrayExtSend.argtypes = (c_int, POINTER(c_int), c_int, c_int, c_char_p, c_int)
    self.CkArrayExtSend = self.lib.CkArrayExtSend
    self.CkGroupExtSend = self.lib.CkGroupExtSend
    self.CkChareExtSend = self.lib.CkChareExtSend

    self.CkMyPe = self.lib.CkMyPeHook
    self.CkNumPes = self.lib.CkNumPesHook
    self.CkExit = self.lib.CkExit

  def CkAbort(self, msg):
    self.lib.CmiAbort(msg.encode())

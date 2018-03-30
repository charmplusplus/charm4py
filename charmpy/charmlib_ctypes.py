import ctypes
from ctypes import c_int, c_short, c_char, c_long, c_uint, c_ushort, c_ubyte, c_ulong, c_float, c_double, c_char_p, c_void_p, POINTER, CFUNCTYPE, Structure, sizeof
import sys
import os
import time
if sys.version_info < (3, 0, 0):
  import cPickle
else:
  import pickle as cPickle
import ckreduction as red
import array
try:
  import numpy
  haveNumpy = True
except ImportError:
  # this is to avoid numpy dependency
  haveNumpy = False
  class NumpyDummyModule:
    class ndarray: pass
    class number: pass
  numpy = NumpyDummyModule()


# Import some useful structures defined on Charm side

### !!! The order of fields here should match the struct CkReductionTypesExt in ckreduction.h !!! ####
class ReducerTypes(Structure):
  _fields_ = [
    ("nop",                 c_int),
    ("sum_char",            c_int),
    ("sum_short",           c_int),
    ("sum_int",             c_int),
    ("sum_long",            c_int),
    ("sum_uchar",           c_int),
    ("sum_ushort",          c_int),
    ("sum_uint",            c_int),
    ("sum_ulong",           c_int),
    ("sum_float",           c_int),
    ("sum_double",          c_int),
    ("product_char",        c_int),
    ("product_short",       c_int),
    ("product_int",         c_int),
    ("product_long",        c_int),
    ("product_uchar",       c_int),
    ("product_ushort",      c_int),
    ("product_uint",        c_int),
    ("product_ulong",       c_int),
    ("product_float",       c_int),
    ("product_double",      c_int),
    ("max_char",            c_int),
    ("max_short",           c_int),
    ("max_int",             c_int),
    ("max_long",            c_int),
    ("max_uchar",           c_int),
    ("max_ushort",          c_int),
    ("max_uint",            c_int),
    ("max_ulong",           c_int),
    ("max_float",           c_int),
    ("max_double",          c_int),
    ("min_char",            c_int),
    ("min_short",           c_int),
    ("min_int",             c_int),
    ("min_long",            c_int),
    ("min_uchar",           c_int),
    ("min_ushort",          c_int),
    ("min_uint",            c_int),
    ("min_ulong",           c_int),
    ("min_float",           c_int),
    ("min_double",          c_int),
    ("external_py",         c_int)
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

  def __init__(self, charm, opts, libcharm_path):
    self.direct_copy_supported = False
    self.name = 'ctypes'
    self.chareNames = []
    self.charm = charm
    self.opts = opts
    self.init(libcharm_path)
    self.ReducerType = ReducerTypes.in_dll(self.lib, "charm_reducers")
    self.times = [0.0] * 3 # track time in [charm reduction callbacks, custom reduction, outgoing object migration]
    self.c_type_table = [None] * 10
    self.c_type_table[red.C_CHAR] = c_char
    self.c_type_table[red.C_SHORT] = c_short
    self.c_type_table[red.C_INT] = c_int
    self.c_type_table[red.C_LONG] = c_long
    self.c_type_table[red.C_UCHAR] = c_ubyte
    self.c_type_table[red.C_USHORT] = c_ushort
    self.c_type_table[red.C_UINT] = c_uint
    self.c_type_table[red.C_ULONG] = c_ulong
    self.c_type_table[red.C_FLOAT] = c_float
    self.c_type_table[red.C_DOUBLE] = c_double
    self.emptyMsg = cPickle.dumps(({},[]))

  def sizeof(self, c_type_id):
    return ctypes.sizeof(self.c_type_table[c_type_id])

  def getReductionTypesFields(self):
    return [field_name for field_name,field_type in ReducerTypes._fields_]

  def initContributeInfo(self, elemId, index, elemType):
    if type(index) == int: index = (index,)
    ndims = len(index)
    c_elemIdx = (c_int*ndims)(*index)
    return ContributeInfo(-1, 0, 0, 0, self.ReducerType.nop, elemId, c_elemIdx,
                          ndims, elemType)

  def getContributeInfo(self, ep, contribution, contributor):
    reducer_type, data, c_type = contribution
    if reducer_type == self.ReducerType.external_py:
      numElems = len(data)
      c_data = c_char_p(data)
      c_data_size = numElems * sizeof(c_char)
    elif reducer_type != self.ReducerType.nop:
      dataType = self.c_type_table[c_type]
      t = type(data)
      if t == numpy.ndarray or isinstance(data, numpy.number):
        numElems = data.size
        c_data = (dataType*numElems).from_buffer(data)  # get pointer to data, no copy
        c_data_size = data.nbytes
      elif t == array.array:
        numElems = len(data)
        c_data = (dataType*numElems).from_buffer(data)  # get pointer to data, no copy
        c_data_size = data.buffer_info()[1] * data.itemsize
      else:
        numElems = len(data)
        c_data = (dataType*numElems)(*data) # this is *really* slow when data is large
        c_data_size = numElems*sizeof(dataType)
    else:
      c_data = None
      c_data_size = numElems = 0

    c_info = contributor._contributeInfo
    c_info.cbEpIdx = ep
    c_info.data = ctypes.cast(c_data, c_void_p)
    c_info.numelems = numElems
    c_info.dataSize = c_data_size
    c_info.redType = reducer_type
    return c_info

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

  def recvChareMsg(self, onPe, objPtr, ep, msgSize, msg, dcopy_start):
    try:
      t0 = None
      if self.opts.PROFILING:
        t0 = time.time()
        self.charm.msg_recv_sizes.append(msgSize)
      if msgSize > 0: msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
      self.charm.recvChareMsg((onPe, objPtr), ep, msg, t0, dcopy_start)
    except:
      self.charm.handleGeneralError()

  def recvGroupMsg(self, gid, ep, msgSize, msg, dcopy_start):
    try:
      t0 = None
      if self.opts.PROFILING:
        t0 = time.time()
        self.charm.msg_recv_sizes.append(msgSize)
      if msgSize > 0: msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
      self.charm.recvGroupMsg(gid, ep, msg, t0, dcopy_start)
    except:
      self.charm.handleGeneralError()

  def recvArrayMsg(self, aid, ndims, arrayIndex, ep, msgSize, msg, dcopy_start):
    try:
      t0 = None
      if self.opts.PROFILING:
        t0 = time.time()
        self.charm.msg_recv_sizes.append(msgSize)
      arrIndex = self.arrayIndexToTuple(ndims, arrayIndex)
      if msgSize > 0: msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
      else: msg = b''
      self.charm.recvArrayMsg(aid, arrIndex, ep, msg, t0, dcopy_start)
    except:
      self.charm.handleGeneralError()

  def CkChareSend(self, chare_id, ep, msg):
    msg0, dcopy = msg
    self.lib.CkChareExtSend(chare_id[0], chare_id[1], ep, msg0, len(msg0))

  def CkGroupSend(self, group_id, index, ep, msg):
    msg0, dcopy = msg
    self.lib.CkGroupExtSend(group_id, index, ep, msg0, len(msg0))

  def CkArraySend(self, array_id, index, ep, msg):
    msg0, dcopy = msg
    ndims = len(index)
    c_elemIdx = (ctypes.c_int * ndims)(*index)  # TODO have buffer preallocated for this?
    self.lib.CkArrayExtSend(array_id, c_elemIdx, ndims, ep, msg0, len(msg0))

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
    if all(v == 0 for v in dims): ndims = -1   # for creating an empty array Charm++ API expects ndims set to -1
    return self.lib.CkCreateArrayExt(chareIdx, ndims, dimsArray, epIdx, None, 0)

  def CkInsert(self, aid, index, epIdx, onPE):
    indexDims = len(index)
    c_index = (c_int*indexDims)(*index)
    self.lib.CkInsertArrayExt(aid, indexDims, c_index, epIdx, onPE, None, 0)

  def CkMigrate(self, aid, index, toPe):
    indexDims = len(index)
    c_index = (c_int*indexDims)(*index)
    self.lib.CkMigrateExt(aid, indexDims, c_index, toPe)

  def CkDoneInserting(self, aid):
    self.lib.CkArrayDoneInsertingExt(aid)

  def getTopoTreeEdges(self, pe, root_pe, pes, bfactor):
    parent       = c_int(0)
    child_count  = c_int(0)
    children_ptr = ctypes.POINTER(ctypes.c_int)()
    if pes is not None:
      pes_c = (c_int*len(pes))(*pes)
      self.lib.getPETopoTreeEdges(pe, root_pe, pes_c, len(pes), bfactor, ctypes.byref(parent),
                                  ctypes.byref(child_count), ctypes.byref(children_ptr))
    else:
      self.lib.getPETopoTreeEdges(pe, root_pe, 0, 0, bfactor, ctypes.byref(parent),
                                  ctypes.byref(child_count), ctypes.byref(children_ptr))
    children = [children_ptr[i] for i in range(int(child_count.value))]
    if len(children) > 0: self.lib.free(children_ptr)
    parent = int(parent.value)
    if parent == -1: parent = None
    return parent, children

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
      if self.opts.PROFILING: t0 = time.time()
      if sizing:
        arrIndex = self.arrayIndexToTuple(ndims, arrayIndex)
        self.tempData = ctypes.create_string_buffer(self.charm.arrayElemLeave(aid, arrIndex))
      else:
        #pdata[0] = ctypes.cast(data, c_void_p).value
        pdata = ctypes.cast(pdata, POINTER(POINTER(c_char)))
        pdata[0] = self.tempData
      if self.opts.PROFILING: self.times[2] += (time.time() - t0)
      return len(self.tempData)
    except:
      self.charm.handleGeneralError()

  def arrayElemJoin(self, aid, ndims, arrayIndex, ep, msg, msgSize):
    try:
      t0 = None
      if self.opts.PROFILING:
        t0 = time.time()
        self.charm.msg_recv_sizes.append(msgSize)
      arrIndex = self.arrayIndexToTuple(ndims, arrayIndex)
      if msgSize > 0: msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
      self.charm.recvArrayMsg(aid, arrIndex, ep, msg, t0, -1)
    except:
      self.charm.handleGeneralError()

  def resumeFromSync(self, aid, ndims, arrayIndex):
    try:
      index = self.arrayIndexToTuple(ndims, arrayIndex)
      self.CkArraySend(aid, index, self.charm.arrays[aid][index].thisProxy.resumeFromSync.ep,
                       (self.emptyMsg, []))
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

  def cpickleData(self, data, dataSize, reducerType, returnBuffers, returnBufferSizes):
    try:
      if self.opts.PROFILING: t0 = time.time()
      header = {}
      if reducerType != self.ReducerType.nop:
        ctype = self.charm.redMgr.charm_reducer_to_ctype[reducerType]
        dataType = self.c_type_table[ctype]
        numElems = dataSize // sizeof(dataType)
        if numElems == 1:
          pyData = [ctypes.cast(data, POINTER(dataType))[0]]
        elif sys.version_info[0] < 3:
          data = ctypes.cast(data, POINTER(dataType * numElems)).contents
          if haveNumpy:
            dt = self.charm.redMgr.rev_np_array_type_map[ctype]
            a = numpy.fromstring(data, dtype=numpy.dtype(dt))
          else:
            array_typecode = self.charm.redMgr.rev_array_type_map[ctype]
            a = array.array(array_typecode, data)
          pyData = [a]
        else:
          if haveNumpy:
            dtype = self.charm.redMgr.rev_np_array_type_map[ctype]
            header[b'dcopy'] = [(0, 2, (numElems, dtype), dataSize)]
          else:
            array_typecode = self.charm.redMgr.rev_array_type_map[ctype]
            header[b'dcopy'] = [(0, 1, (array_typecode), dataSize)]
          returnBuffers[1] = ctypes.cast(data, c_char_p)
          returnBufferSizes[1] = dataSize
          pyData = [None]
      else:
        pyData = []

      msg = (header, pyData)
      pickledData = cPickle.dumps(msg, self.opts.PICKLE_PROTOCOL)
      returnBuffers = ctypes.cast(returnBuffers, POINTER(POINTER(c_char)))
      returnBuffers[0] = ctypes.cast(c_char_p(pickledData), POINTER(c_char))
      returnBufferSizes[0] = len(pickledData)

      if self.opts.PROFILING: self.times[0] += (time.time() - t0)
    except:
      self.charm.handleGeneralError()

  # callback function invoked by Charm++ for reducing contributions using a Python reducer (built-in or custom)
  def pyReduction(self, msgs, msgSizes, nMsgs, returnBuffer):
    try:
      if self.opts.PROFILING: t0 = time.time()
      contribs = []
      currentReducer = None
      for i in range(nMsgs):
        msg = msgs[i]
        if msgSizes[i] > 0:
          if self.opts.PROFILING: self.charm.msg_recv_sizes.append(int(msgSizes[i]))
          msg = ctypes.cast(msg, POINTER(c_char * msgSizes[i])).contents.raw
          header, args = cPickle.loads(msg)

          customReducer = header[b"custom_reducer"]

          if currentReducer is None: currentReducer = customReducer
          # check for correctness of msg
          assert customReducer == currentReducer

          contribs.append(args[0])

      reductionResult = getattr(self.charm.reducers, currentReducer)(contribs)
      rednMsg = ({b"custom_reducer": currentReducer}, [reductionResult])
      rednMsgPickle = cPickle.dumps(rednMsg, self.opts.PICKLE_PROTOCOL)
      rednMsgPickle = ctypes.create_string_buffer(rednMsgPickle)

      # cast returnBuffer to char** and make it point to pickled reduction msg
      returnBuffer = ctypes.cast(returnBuffer, POINTER(POINTER(c_char)))
      returnBuffer[0] = rednMsgPickle

      if self.opts.PROFILING: self.times[1] += (time.time() - t0)
      return len(rednMsgPickle)
    except:
      self.charm.handleGeneralError()

  # first callback from Charm++ shared library
  def registerMainModule(self):
    try:
      self.charm.registerMainModule()
    except:
      self.charm.handleGeneralError()

  def lib_version_check(self):
    commit_id = ctypes.c_char_p.in_dll(self.lib, "CmiCommitID").value.decode()
    self.charm.lib_version_check(commit_id)

  def init(self, libcharm_path):
    p = os.environ.get("LIBCHARM_PATH")
    if p is not None: libcharm_path = p
    if libcharm_path != None:
      self.lib = ctypes.CDLL(libcharm_path + '/libcharm.so')
    else:
      self.lib = ctypes.CDLL("libcharm.so")

    self.lib_version_check()

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

    self.RECV_CHARE_CB_TYPE = CFUNCTYPE(None, c_int, c_void_p, c_int, c_int, POINTER(c_char), c_int)
    self.recvChareCb = self.RECV_CHARE_CB_TYPE(self.recvChareMsg)
    self.lib.registerChareMsgRecvExtCallback(self.recvChareCb)

    self.RECV_GROUP_CB_TYPE = CFUNCTYPE(None, c_int, c_int, c_int, POINTER(c_char), c_int)
    self.recvGroupCb = self.RECV_GROUP_CB_TYPE(self.recvGroupMsg)
    self.lib.registerGroupMsgRecvExtCallback(self.recvGroupCb)

    self.RECV_ARRAY_CB_TYPE = CFUNCTYPE(None, c_int, c_int, POINTER(c_int), c_int, c_int, POINTER(c_char), c_int)
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
    self.CPICKLE_DATA_CB_TYPE = CFUNCTYPE(None, c_void_p, c_int, c_int, POINTER(c_char_p), POINTER(c_int))
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

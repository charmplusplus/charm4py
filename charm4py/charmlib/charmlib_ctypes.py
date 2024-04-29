import ctypes
from ctypes import c_int, c_short, c_char, c_long, c_longlong, c_byte, \
                   c_uint, c_ushort, c_ubyte, c_ulong, c_ulonglong, \
                   c_float, c_double, c_char_p, c_void_p, POINTER, CFUNCTYPE, Structure, sizeof
import sys
import os
import time
import platform
if sys.version_info < (3, 0, 0):
  import cPickle
else:
  import pickle as cPickle
from .. import reduction as red
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
    ("sum_long_long",       c_int),
    ("sum_uchar",           c_int),
    ("sum_ushort",          c_int),
    ("sum_uint",            c_int),
    ("sum_ulong",           c_int),
    ("sum_ulong_long",      c_int),
    ("sum_float",           c_int),
    ("sum_double",          c_int),
    ("product_char",        c_int),
    ("product_short",       c_int),
    ("product_int",         c_int),
    ("product_long",        c_int),
    ("product_long_long",   c_int),
    ("product_uchar",       c_int),
    ("product_ushort",      c_int),
    ("product_uint",        c_int),
    ("product_ulong",       c_int),
    ("product_ulong_long",  c_int),
    ("product_float",       c_int),
    ("product_double",      c_int),
    ("max_char",            c_int),
    ("max_short",           c_int),
    ("max_int",             c_int),
    ("max_long",            c_int),
    ("max_long_long",       c_int),
    ("max_uchar",           c_int),
    ("max_ushort",          c_int),
    ("max_uint",            c_int),
    ("max_ulong",           c_int),
    ("max_ulong_long",      c_int),
    ("max_float",           c_int),
    ("max_double",          c_int),
    ("min_char",            c_int),
    ("min_short",           c_int),
    ("min_int",             c_int),
    ("min_long",            c_int),
    ("min_long_long",       c_int),
    ("min_uchar",           c_int),
    ("min_ushort",          c_int),
    ("min_uint",            c_int),
    ("min_ulong",           c_int),
    ("min_ulong_long",      c_int),
    ("min_float",           c_int),
    ("min_double",          c_int),
    ("logical_and_bool",    c_int),
    ("logical_or_bool",     c_int),
    ("logical_xor_bool",    c_int),
    ("external_py",         c_int)
  ]

class ContributeInfo(Structure):
  _fields_ = [
    ("cbEpIdx", c_int),               # index of entry point at reduction target
    ("fid", c_int),                   # future ID (used when reduction target is a future)
    ("data", c_void_p),               # data contributed for reduction
    ("numelems", c_int),              # number of elements in data
    ("dataSize", c_int),              # size of data in bytes
    ("redType", c_int),               # type of reduction (ReducerTypes)
    ("id", c_int),                    # ID of the contributing array/group
    ("idx", POINTER(c_int)),          # index of the contributing chare array/group element
    ("ndims", c_int),                 # number of dimensions in index
    ("contributorType", c_int)        # type of contributor
  ]

  def getDataSize(self):
    return self.dataSize

class CharmLib(object):

  def __init__(self, charm, opts, libcharm_path):
    self.direct_copy_supported = False
    self.name = 'ctypes'
    self.chareNames = []
    self.charm = charm
    self.opts = opts
    self.system = platform.system().lower()
    self.init(libcharm_path)
    self.ReducerType = ReducerTypes.in_dll(self.lib, "charm_reducers")
    self.times = [0.0] * 3 # track time in [charm reduction callbacks, custom reduction, outgoing object migration]
    self.c_type_table = [None] * 13
    self.c_type_table[red.C_BOOL] = c_byte
    self.c_type_table[red.C_CHAR] = c_char
    self.c_type_table[red.C_SHORT] = c_short
    self.c_type_table[red.C_INT] = c_int
    self.c_type_table[red.C_LONG] = c_long
    self.c_type_table[red.C_LONG_LONG] = c_longlong
    self.c_type_table[red.C_UCHAR] = c_ubyte
    self.c_type_table[red.C_USHORT] = c_ushort
    self.c_type_table[red.C_UINT] = c_uint
    self.c_type_table[red.C_ULONG] = c_ulong
    self.c_type_table[red.C_ULONG_LONG] = c_ulonglong
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
    return ContributeInfo(-1, 0, 0, 0, 0, self.ReducerType.nop, elemId, c_elemIdx,
                          ndims, elemType)

  def getContributeInfo(self, ep, fid, contribution, contributor):
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
    c_info.fid = fid
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
      if self.opts.profiling:
        self.charm._precvtime = time.time()
        self.charm.recordReceive(msgSize)
      if msgSize > 0: msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
      self.charm.recvChareMsg((onPe, objPtr), ep, msg, dcopy_start)
    except:
      self.charm.handleGeneralError()

  def recvGroupMsg(self, gid, ep, msgSize, msg, dcopy_start):
    try:
      if self.opts.profiling:
        self.charm._precvtime = time.time()
        self.charm.recordReceive(msgSize)
      if msgSize > 0: msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
      self.charm.recvGroupMsg(gid, ep, msg, dcopy_start)
    except:
      self.charm.handleGeneralError()

  def recvArrayMsg(self, aid, ndims, arrayIndex, ep, msgSize, msg, dcopy_start):
    try:
      if self.opts.profiling:
        self.charm._precvtime = time.time()
        self.charm.recordReceive(msgSize)
      arrIndex = self.arrayIndexToTuple(ndims, arrayIndex)
      if msgSize > 0: msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
      else: msg = b''
      self.charm.recvArrayMsg(aid, arrIndex, ep, msg, dcopy_start)
    except:
      self.charm.handleGeneralError()

  def recvArrayBcast(self, aid, ndims, nInts, numElems, arrayIndexes, ep, msgSize, msg, dcopy_start):
    try:
      if self.opts.profiling:
        self.charm._precvtime = time.time()
        self.charm.recordReceive(msgSize)
      indexes = []
      arrayIndexes_p_val = ctypes.cast(arrayIndexes, c_void_p).value
      sizeof_int = sizeof(c_int)
      for i in range(numElems):
        indexes.append(self.arrayIndexToTuple(ndims, arrayIndexes_p_val + sizeof_int * (i*nInts)))
      if msgSize > 0: msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
      else: msg = b''
      self.charm.recvArrayBcast(aid, indexes, ep, msg, dcopy_start)
    except:
      self.charm.handleGeneralError()

  def arrayMapProcNum(self, gid, ndims, arrayIndex):
    try:
      arrIndex = self.arrayIndexToTuple(ndims, arrayIndex)
      return self.charm.arrayMapProcNum(gid, arrIndex)
    except:
      self.charm.handleGeneralError()

  def CkChareSend(self, chare_id, ep, msg):
    msg0, dcopy = msg
    self.lib.CkChareExtSend(chare_id[0], chare_id[1], ep, msg0, len(msg0))

  def CkGroupSend(self, group_id, index, ep, msg):
    msg0, dcopy = msg
    c_pe = c_int(index)
    self.lib.CkGroupExtSend(group_id, 1, ctypes.byref(c_pe), ep, msg0, len(msg0))

  def CkGroupSendMulti(self, group_id, pes, ep, msg):
    msg0, dcopy = msg
    c_pes = (c_int * len(pes))(*pes)
    self.lib.CkGroupExtSend(group_id, len(pes), c_pes, ep, msg0, len(msg0))

  def CkArraySend(self, array_id, index, ep, msg):
    msg0, dcopy = msg
    ndims = len(index)
    c_elemIdx = (c_int * ndims)(*index)  # TODO have buffer preallocated for this?
    self.lib.CkArrayExtSend(array_id, c_elemIdx, ndims, ep, msg0, len(msg0))

  def sendToSection(self, gid, children):
    c_children = (c_int * len(children))(*children)
    self.lib.CkForwardMulticastMsg(gid, len(children), c_children)

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

  def CkRegisterSectionManager(self, name, numEntryMethods):
    self.chareNames.append(ctypes.create_string_buffer(name.encode()))
    chareIdx, startEpIdx = c_int(0), c_int(0)
    self.lib.CkRegisterSectionManagerExt(self.chareNames[-1], numEntryMethods, ctypes.byref(chareIdx), ctypes.byref(startEpIdx))
    return int(chareIdx.value), int(startEpIdx.value)

  def CkRegisterArrayMap(self, name, numEntryMethods):
    self.chareNames.append(ctypes.create_string_buffer(name.encode()))
    chareIdx, startEpIdx = c_int(0), c_int(0)
    self.lib.CkRegisterArrayMapExt(self.chareNames[-1], numEntryMethods, ctypes.byref(chareIdx), ctypes.byref(startEpIdx))
    return int(chareIdx.value), int(startEpIdx.value)

  def CkRegisterArray(self, name, numEntryMethods):
    self.chareNames.append(ctypes.create_string_buffer(name.encode()))
    chareIdx, startEpIdx = c_int(0), c_int(0)
    self.lib.CkRegisterArrayExt(self.chareNames[-1], numEntryMethods, ctypes.byref(chareIdx), ctypes.byref(startEpIdx))
    return int(chareIdx.value), int(startEpIdx.value)

  def CkCreateGroup(self, chareIdx, epIdx, msg):
    msg0, dcopy = msg
    msgLenArray = (c_int*1)(len(msg0))
    msgArray = (c_char_p*1)(msg0)
    return self.lib.CkCreateGroupExt(chareIdx, epIdx, 1, msgArray, msgLenArray)

  def CkCreateArray(self, chareIdx, dims, epIdx, msg, map_gid, useAtSync):
    msg0, dcopy = msg
    ndims = len(dims)
    dimsArray = (c_int*ndims)(*dims)
    msgLenArray = (c_int*1)(len(msg0))
    msgArray = (c_char_p*1)(msg0)
    if all(v == 0 for v in dims): ndims = -1   # for creating an empty array Charm++ API expects ndims set to -1
    return self.lib.CkCreateArrayExt(chareIdx, ndims, dimsArray, epIdx, 1,
                                     msgArray, msgLenArray, map_gid, useAtSync)

  def CkInsert(self, aid, index, epIdx, onPE, msg, useAtSync):
    msg0, dcopy = msg
    indexDims = len(index)
    c_index = (c_int*indexDims)(*index)
    msgLenArray = (c_int*1)(len(msg0))
    msgArray = (c_char_p*1)(msg0)
    self.lib.CkInsertArrayExt(aid, indexDims, c_index, epIdx, onPE, 1,
                              msgArray, msgLenArray, useAtSync)

  def CkMigrate(self, aid, index, toPe):
    indexDims = len(index)
    c_index = (c_int*indexDims)(*index)
    self.lib.CkMigrateExt(aid, indexDims, c_index, toPe)

  def CkDoneInserting(self, aid):
    self.lib.CkArrayDoneInsertingExt(aid)

  def getGroupRedNo(self, gid):
    return self.lib.CkGroupGetReductionNumber(gid)

  def getArrayElementRedNo(self, aid, index):
    indexDims = len(index)
    c_index = (c_int*indexDims)(*index)
    return self.lib.CkArrayGetReductionNumber(aid, indexDims, c_index)

  def setMigratable(self, aid, index, migratable):
    ndims = len(index)
    c_index = (c_int*ndims)(*index)
    self.lib.CkSetMigratable(aid, ndims, c_index, migratable)

  def getTopoTreeEdges(self, pe, root_pe, pes, bfactor):
    parent       = c_int(0)
    child_count  = c_int(0)
    children_ptr = ctypes.POINTER(ctypes.c_int)()
    if pes is not None:
      pes_c = (c_int * len(pes))(*pes)
      self.lib.getPETopoTreeEdges(pe, root_pe, pes_c, len(pes), bfactor, ctypes.byref(parent),
                                  ctypes.byref(child_count), ctypes.byref(children_ptr))
    else:
      self.lib.getPETopoTreeEdges(pe, root_pe, 0, 0, bfactor, ctypes.byref(parent),
                                  ctypes.byref(child_count), ctypes.byref(children_ptr))
    children = [children_ptr[i] for i in range(int(child_count.value))]
    if len(children) > 0:
      self.lib.free(children_ptr)
    parent = int(parent.value)
    if parent == -1: parent = None
    return parent, children

  def getTopoSubtrees(self, root_pe, pes, bfactor):
    parent       = c_int(0)
    child_count  = c_int(0)
    children_ptr = ctypes.POINTER(ctypes.c_int)()
    pes_c = (c_int * len(pes))(*pes)
    self.lib.getPETopoTreeEdges(root_pe, root_pe, pes_c, len(pes), bfactor, ctypes.byref(parent),
                                ctypes.byref(child_count), ctypes.byref(children_ptr))
    children = [children_ptr[i] for i in range(int(child_count.value))]
    child_count = len(children)
    if child_count > 0:
      self.lib.free(children_ptr)

    subtrees = []
    if child_count > 0:
      idx = 1
      for i in range(child_count):
        subtree = []
        assert pes_c[idx] == children[i]
        if i < child_count - 1:
          next_child = children[i+1]
        else:
          next_child = None
        while idx < len(pes):
          pe = int(pes_c[idx])
          if pe == next_child:
            break
          subtree.append(pe)
          idx += 1
        subtrees.append(subtree)
    return subtrees

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
      if self.opts.profiling: t0 = time.time()
      if sizing:
        arrIndex = self.arrayIndexToTuple(ndims, arrayIndex)
        self.tempData = ctypes.create_string_buffer(self.charm.arrayElemLeave(aid, arrIndex))
      else:
        #pdata[0] = ctypes.cast(data, c_void_p).value
        pdata = ctypes.cast(pdata, POINTER(POINTER(c_char)))
        pdata[0] = self.tempData
      if self.opts.profiling: self.times[2] += (time.time() - t0)
      return len(self.tempData)
    except:
      self.charm.handleGeneralError()

  def arrayElemJoin(self, aid, ndims, arrayIndex, ep, msg, msgSize):
    try:
      if self.opts.profiling:
        self.charm._precvtime = time.time()
        self.charm.recordReceive(msgSize)
      arrIndex = self.arrayIndexToTuple(ndims, arrayIndex)
      if msgSize > 0: msg = ctypes.cast(msg, POINTER(c_char * msgSize)).contents.raw
      self.charm.recvArrayMsg(aid, arrIndex, ep, msg, -1)
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

  def CkContributeToSection(self, contributeInfo, sid, rootPE):
    self.lib.CkExtContributeToSection(ctypes.byref(contributeInfo), sid[0], sid[1], rootPE)

  def CkStartQD_ChareCallback(self, cid, ep, fid):
    self.lib.CkStartQDExt_ChareCallback(cid[0], cid[1], ep, fid)

  def CkStartQD_GroupCallback(self, gid, pe, ep, fid):
    self.lib.CkStartQDExt_GroupCallback(gid, pe, ep, fid)

  def CkStartQD_ArrayCallback(self, aid, index, ep, fid):
    ndims = len(index)
    c_index = (ctypes.c_int * ndims)(*index)
    self.lib.CkStartQDExt_ArrayCallback(aid, c_index, ndims, ep, fid)

  def CkStartQD_SectionCallback(self, sid, rootPE, ep):
    self.lib.CkStartQDExt_SectionCallback(sid[0], sid[1], rootPE, ep)

  def createCallbackMsg(self, data, dataSize, reducerType, fid, sectionInfo, returnBuffers, returnBufferSizes):
    try:
      if self.opts.profiling: t0 = time.time()

      pyData = []
      if sectionInfo[0] >= 0:
        # this is a section callback
        sid = (sectionInfo[0], sectionInfo[1])
        pyData = [sid, sectionInfo[2], {b'sid': sid}]
        secMgrProxy = self.charm.sectionMgr.thisProxy
        # tell Charm++ the gid and ep of SectionManager for section broadcasts
        sectionInfo[0] = secMgrProxy.gid
        sectionInfo[1] = secMgrProxy.sendToSection.ep

      if (reducerType < 0) or (reducerType == self.ReducerType.nop):
        if fid > 0:
          msg = ({}, [fid])
        else:
          msg = ({}, pyData)
        pickledData = cPickle.dumps(msg, self.opts.pickle_protocol)
        returnBuffers = ctypes.cast(returnBuffers, POINTER(POINTER(c_char)))
        returnBuffers[0] = ctypes.cast(c_char_p(pickledData), POINTER(c_char))
        returnBufferSizes[0] = len(pickledData)

      elif reducerType != self.ReducerType.external_py:
        header = {}
        ctype = self.charm.redMgr.charm_reducer_to_ctype[reducerType]
        dataType = self.c_type_table[ctype]
        numElems = dataSize // sizeof(dataType)
        if fid > 0:
          pyData.append(fid)
        if numElems == 1:
          pyData.append(ctypes.cast(data, POINTER(dataType))[0])
        elif sys.version_info[0] < 3:
          data = ctypes.cast(data, POINTER(dataType * numElems)).contents
          if haveNumpy:
            dt = self.charm.redMgr.rev_np_array_type_map[ctype]
            a = numpy.frombuffer(data, dtype=numpy.dtype(dt))
          else:
            array_typecode = self.charm.redMgr.rev_array_type_map[ctype]
            a = array.array(array_typecode, data)
          pyData.append(a)
        else:
          if haveNumpy:
            dtype = self.charm.redMgr.rev_np_array_type_map[ctype]
            header[b'dcopy'] = [(len(pyData), 2, (numElems, dtype), dataSize)]
          else:
            array_typecode = self.charm.redMgr.rev_array_type_map[ctype]
            header[b'dcopy'] = [(len(pyData), 1, (array_typecode), dataSize)]
          returnBuffers[1] = ctypes.cast(data, c_char_p)
          returnBufferSizes[1] = dataSize
          pyData.append(None)

        msg = (header, pyData)
        pickledData = cPickle.dumps(msg, self.opts.pickle_protocol)
        returnBuffers = ctypes.cast(returnBuffers, POINTER(POINTER(c_char)))
        returnBuffers[0] = ctypes.cast(c_char_p(pickledData), POINTER(c_char))
        returnBufferSizes[0] = len(pickledData)

      elif fid > 0 or len(pyData) > 0:
        # TODO: this is INEFFICIENT. it unpickles the message, then either:
        # a) inserts the future ID as first argument
        # b) puts the data into a section msg
        # then repickles the message.
        # this code path is only used when the result of a reduction using a
        # Python-defined (custom) reducer is sent to a Future or Section.
        # If this turns out to be critical we should consider a more efficient solution
        data = ctypes.cast(data, POINTER(c_char * dataSize)).contents.raw
        header, args = cPickle.loads(data)
        if fid > 0:
          args.insert(0, fid)
        else:
          pyData.extend(args)
          args = pyData
        pickledData = cPickle.dumps((header,args), self.opts.pickle_protocol)
        returnBuffers = ctypes.cast(returnBuffers, POINTER(POINTER(c_char)))
        returnBuffers[0] = ctypes.cast(c_char_p(pickledData), POINTER(c_char))
        returnBufferSizes[0] = len(pickledData)
      else:
        # do nothing, use message as is (was created by Charm4py)
        returnBuffers[0]     = ctypes.cast(data, c_char_p)
        returnBufferSizes[0] = dataSize

      if self.opts.profiling: self.times[0] += (time.time() - t0)
    except:
      self.charm.handleGeneralError()

  # callback function invoked by Charm++ for reducing contributions using a Python reducer (built-in or custom)
  def pyReduction(self, msgs, msgSizes, nMsgs, returnBuffer):
    try:
      if self.opts.profiling: t0 = time.time()
      contribs = []
      currentReducer = None
      for i in range(nMsgs):
        msg = msgs[i]
        if msgSizes[i] > 0:
          if self.opts.profiling: self.charm.recordReceive(int(msgSizes[i]))
          msg = ctypes.cast(msg, POINTER(c_char * msgSizes[i])).contents.raw
          header, args = cPickle.loads(msg)

          customReducer = header[b"custom_reducer"]

          if currentReducer is None: currentReducer = customReducer
          # check for correctness of msg
          assert customReducer == currentReducer

          contribs.append(args[0])

      reductionResult = getattr(self.charm.reducers, currentReducer)(contribs)
      rednMsg = ({b"custom_reducer": currentReducer}, [reductionResult])
      rednMsgPickle = cPickle.dumps(rednMsg, self.opts.pickle_protocol)
      rednMsgPickle = ctypes.create_string_buffer(rednMsgPickle)

      # cast returnBuffer to char** and make it point to pickled reduction msg
      returnBuffer = ctypes.cast(returnBuffer, POINTER(POINTER(c_char)))
      returnBuffer[0] = rednMsgPickle

      if self.opts.profiling: self.times[1] += (time.time() - t0)
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
    commit_id = ctypes.c_char_p.in_dll(self.lib, 'CmiCommitID').value.decode()
    self.charm.lib_version_check(commit_id)

  def init(self, libcharm_path):
    import os.path
    if os.name != 'nt':
      p = os.environ.get('LIBCHARM_PATH')
      if p is not None: libcharm_path = p
      if libcharm_path is not None:
        if self.system == 'darwin':
          self.lib = ctypes.CDLL(os.path.join(libcharm_path, 'libcharm.dylib'))
        else:
          self.lib = ctypes.CDLL(os.path.join(libcharm_path, 'libcharm.so'))
      else:
        if self.system == 'darwin':
          self.lib = ctypes.CDLL('libcharm.dylib')
        else:
          self.lib = ctypes.CDLL('libcharm.so')
    else:
      self.lib = ctypes.CDLL(os.path.join(libcharm_path, 'charm.dll'))

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

    self.RECV_ARRAY_BCAST_CB_TYPE = CFUNCTYPE(None, c_int, c_int, c_int, c_int, POINTER(c_int), c_int, c_int, POINTER(c_char), c_int)
    self.recvArrayBcastCb = self.RECV_ARRAY_BCAST_CB_TYPE(self.recvArrayBcast)
    self.lib.registerArrayBcastRecvExtCallback(self.recvArrayBcastCb)

    self.ARRAY_MAP_PROCNUM_CB_TYPE = CFUNCTYPE(c_int, c_int, c_int, POINTER(c_int))
    self.arrayMapProcNumCb = self.ARRAY_MAP_PROCNUM_CB_TYPE(self.arrayMapProcNum)
    self.lib.registerArrayMapProcNumExtCallback(self.arrayMapProcNumCb)

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

    # Args to createCallbackMsg: data, return_buffer, data_size, reducer_type
    self.CREATE_RED_TARG_MSG_CB_TYPE = CFUNCTYPE(None, c_void_p, c_int, c_int, c_int, POINTER(c_int), POINTER(c_char_p), POINTER(c_int))
    self.createCallbackMsgCb = self.CREATE_RED_TARG_MSG_CB_TYPE(self.createCallbackMsg)
    self.lib.registerCreateCallbackMsgExtCallback(self.createCallbackMsgCb)

    # Args to pyReduction: msgs, msgSizes, nMsgs, returnBuffer
    self.PY_REDUCTION_CB_TYPE = CFUNCTYPE(c_int, POINTER(c_void_p), POINTER(c_int), c_int, POINTER(c_char_p))
    self.pyReductionCb = self.PY_REDUCTION_CB_TYPE(self.pyReduction)
    self.lib.registerPyReductionExtCallback(self.pyReductionCb)

    # the following line decreases performance, don't know why. seems to work fine without it
    #self.lib.CkArrayExtSend.argtypes = (c_int, POINTER(c_int), c_int, c_int, c_char_p, c_int)
    self.lib.CkChareExtSend.argtypes = (c_int, c_void_p, c_int, c_char_p, c_int)
    self.CkArrayExtSend = self.lib.CkArrayExtSend
    self.CkGroupExtSend = self.lib.CkGroupExtSend
    self.CkChareExtSend = self.lib.CkChareExtSend
    self.lib.CkExtContributeToChare.argtypes = (c_void_p, c_int, c_void_p)
    self.lib.CkStartQDExt_ChareCallback.argtypes = (c_int, c_void_p, c_int, c_int)

    self.CcdCallFnAfterCallback_cb = CFUNCTYPE(None, c_void_p, c_double)(self.CcdCallFnAfterCallback)

    self.CkMyPe = self.lib.CkMyPeHook
    self.CkNumPes = self.lib.CkNumPesHook
    self.CkExit = self.lib.realCkExit
    self.CkPrintf = self.lib.CmiPrintf

  def CkAbort(self, msg):
    self.lib.CmiAbort(b"%s", msg.encode())

  def LBTurnInstrumentOn(self):
    self.lib.LBTurnInstrumentOn()

  def LBTurnInstrumentOff(self):
    self.lib.LBTurnInstrumentOff()

  def CkGetFirstPeOnPhysicalNode(self, node):
    return self.lib.CmiGetFirstPeOnPhysicalNode(node)

  def CkPhysicalNodeID(self, pe):
    return self.lib.CmiPhysicalNodeID(pe)

  def CkNumPhysicalNodes(self):
    return self.lib.CmiNumPhysicalNodes()

  def CkNumPesOnPhysicalNode(self, node):
    return self.lib.CmiNumPesOnPhysicalNode(node)

  def CkPhysicalRank(self, pe):
    return self.lib.CmiPhysicalRank(pe)

  def CkGetPesOnPhysicalNode(self, node):
    numpes = c_int(0)
    pelist = POINTER(c_int)()
    self.lib.CmiGetPesOnPhysicalNode(node, ctypes.byref(pelist), ctypes.byref(numpes))
    return [pelist[i] for i in range(numpes.value)]

  def scheduleTagAfter(self, tag, msecs):
    self.lib.CcdCallFnAfter(self.CcdCallFnAfterCallback_cb, tag, c_double(msecs))

  def CcdCallFnAfterCallback(self, userParam, curWallTime):
    try:
      self.charm.triggerCallable(userParam)
    except:
      self.charm.handleGeneralError()

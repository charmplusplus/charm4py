from ._charmlib_cffi import ffi, lib
import sys
import time
if sys.version_info[0] < 3:
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


index_ctype = ('', 'int[1]', 'int[2]', 'int[3]', 'short[4]', 'short[5]', 'short[6]')
emptyMsg = cPickle.dumps(({},[]))

class ContributeInfo:
  def __init__(self, args):
    # Need to save these cdata objects or they will be deleted. Simply putting them
    # in the 'struct ContributeInfo' is not enough
    self.c_data = args[2]
    self.dataSize = args[4]
    self.c_idx  = args[7]
    self.data = ffi.new("struct ContributeInfo*", args)

  def getDataSize(self):
    return self.dataSize

class CharmLib(object):

  def __init__(self, _charm, opts, libcharm_path):
    global charm, ReducerType, c_type_table, times
    self.direct_copy_supported = (sys.version_info[0] >= 3) # requires Python 3
    charm = _charm
    self.name = 'cffi'
    self.chareNames = []
    self.init()
    ReducerType = ffi.cast('struct CkReductionTypesExt*', lib.getReducersStruct())
    self.ReducerType = ReducerType
    times = [0.0] * 3 # track time in [charm reduction callbacks, custom reduction, outgoing object migration]
    self.times = times
    self.send_bufs = ffi.new("char*[]", 60)  # supports up to 60 direct-copy entry method arguments
    self.send_buf_sizes = ffi.new("int[]", [0] * 60)
    c_type_table = [None] * 12
    c_type_table[red.C_CHAR] = ('char', 'char[]', 'char*', ffi.sizeof('char'))
    c_type_table[red.C_SHORT] = ('short', 'short[]', 'short*', ffi.sizeof('short'))
    c_type_table[red.C_INT] = ('int', 'int[]', 'int*', ffi.sizeof('int'))
    c_type_table[red.C_LONG] = ('long', 'long[]', 'long*', ffi.sizeof('long'))
    c_type_table[red.C_LONG_LONG] = ('long long', 'long long[]', 'long long*', ffi.sizeof('long long'))
    c_type_table[red.C_UCHAR] = ('unsigned char', 'unsigned char[]', 'unsigned char*', ffi.sizeof('unsigned char'))
    c_type_table[red.C_USHORT] = ('unsigned short', 'unsigned short[]', 'unsigned short*', ffi.sizeof('unsigned short'))
    c_type_table[red.C_UINT] = ('unsigned int', 'unsigned int[]', 'unsigned int*', ffi.sizeof('unsigned int'))
    c_type_table[red.C_ULONG] = ('unsigned long', 'unsigned long[]', 'unsigned long*', ffi.sizeof('unsigned long'))
    c_type_table[red.C_ULONG_LONG] = ('unsigned long long', 'unsigned long long[]', 'unsigned long long*', ffi.sizeof('unsigned long long'))
    c_type_table[red.C_FLOAT] = ('float', 'float[]', 'float*', ffi.sizeof('float'))
    c_type_table[red.C_DOUBLE] = ('double', 'double[]', 'double*', ffi.sizeof('double'))

  def sizeof(self, c_type_id):
    return c_type_table[c_type_id][3]

  def getReductionTypesFields(self):
    return [f for (f,t) in ffi.typeof("struct CkReductionTypesExt").fields]

  def initContributeInfo(self, elemId, index, elemType):
    if type(index) == int: index = (index,)
    c_elemIdx = ffi.new('int[]', index)
    return ContributeInfo((-1, 0, ffi.NULL, 0, 0, self.ReducerType.nop, elemId,
                          c_elemIdx, len(index), elemType))

  def getContributeInfo(self, ep, fid, contribution, contributor):
    reducer_type, data, c_type = contribution
    if reducer_type == self.ReducerType.external_py:
      numElems = len(data)
      c_data = ffi.from_buffer(data)  # this avoids a copy
      c_data_size = numElems * ffi.sizeof('char')
    elif reducer_type != self.ReducerType.nop:
      t = type(data)
      if t == numpy.ndarray or isinstance(data, numpy.number):
        c_data = ffi.from_buffer(data)  # get pointer to data, no copy
        c_data_size = data.nbytes
        numElems = data.size
      elif t == array.array:
        c_data = ffi.from_buffer(data)  # get pointer to data, no copy
        c_data_size = data.buffer_info()[1] * data.itemsize
        numElems = len(data)
      else:
        dataTypeTuple = c_type_table[c_type]
        numElems = len(data)
        # this copies and convert data to C array of C type
        c_data = ffi.new(dataTypeTuple[1], data)
        c_data_size = numElems * dataTypeTuple[3]
    else:
      c_data = ffi.NULL
      c_data_size = numElems = 0

    c_info = contributor._contributeInfo
    c_struct = c_info.data
    c_struct.cbEpIdx = ep
    c_struct.fid = fid
    c_struct.data = c_info.c_data = c_data
    c_struct.numelems = numElems
    c_struct.dataSize = c_info.dataSize = c_data_size
    c_struct.redType = reducer_type
    return c_info

  @ffi.def_extern()
  def recvReadOnly_py2(msgSize, msg):
    try:
      charm.recvReadOnly(ffi.buffer(msg, msgSize)[:])
    except:
      charm.handleGeneralError()

  @ffi.def_extern()
  def recvReadOnly_py3(msgSize, msg):
    try:
      charm.recvReadOnly(ffi.buffer(msg, msgSize))
    except:
      charm.handleGeneralError()

  @ffi.def_extern()
  def buildMainchare(onPe, objPtr, ep, argc, argv):
    try:
      objPtr = int(ffi.cast("uintptr_t", objPtr))
      charm.buildMainchare(onPe, objPtr, ep, [ffi.string(argv[i]).decode() for i in range(argc)])
    except:
      charm.handleGeneralError()

  @ffi.def_extern()
  def recvChareMsg_py2(onPe, objPtr, ep, msgSize, msg, dcopy_start):
    try:
      t0 = None
      if charm.opts.PROFILING:
        t0 = time.time()
        charm.recordReceive(msgSize)
      objPtr = int(ffi.cast("uintptr_t", objPtr))
      charm.recvChareMsg((onPe, objPtr), ep, ffi.buffer(msg, msgSize)[:], t0, dcopy_start)
    except:
      charm.handleGeneralError()

  @ffi.def_extern()
  def recvChareMsg_py3(onPe, objPtr, ep, msgSize, msg, dcopy_start):
    try:
      t0 = None
      if charm.opts.PROFILING:
        t0 = time.time()
        charm.recordReceive(msgSize)
      objPtr = int(ffi.cast("uintptr_t", objPtr))
      charm.recvChareMsg((onPe, objPtr), ep, ffi.buffer(msg, msgSize), t0, dcopy_start)
    except:
      charm.handleGeneralError()

  @ffi.def_extern()
  def recvGroupMsg_py2(gid, ep, msgSize, msg, dcopy_start):
    try:
      t0 = None
      if charm.opts.PROFILING:
        t0 = time.time()
        charm.recordReceive(msgSize)
      charm.recvGroupMsg(gid, ep, ffi.buffer(msg, msgSize)[:], t0, dcopy_start)
    except:
      charm.handleGeneralError()

  @ffi.def_extern()
  def recvGroupMsg_py3(gid, ep, msgSize, msg, dcopy_start):
    try:
      t0 = None
      if charm.opts.PROFILING:
        t0 = time.time()
        charm.recordReceive(msgSize)
      charm.recvGroupMsg(gid, ep, ffi.buffer(msg, msgSize), t0, dcopy_start)
    except:
      charm.handleGeneralError()

  @ffi.def_extern()
  def recvArrayMsg_py2(aid, ndims, arrayIndex, ep, msgSize, msg, dcopy_start):
    try:
      t0 = None
      if charm.opts.PROFILING:
        t0 = time.time()
        charm.recordReceive(msgSize)
      arrIndex = tuple(ffi.cast(index_ctype[ndims], arrayIndex))
      charm.recvArrayMsg(aid, arrIndex, ep, ffi.buffer(msg, msgSize)[:], t0, dcopy_start)
    except:
      charm.handleGeneralError()

  @ffi.def_extern()
  def recvArrayMsg_py3(aid, ndims, arrayIndex, ep, msgSize, msg, dcopy_start):
    try:
      t0 = None
      if charm.opts.PROFILING:
        t0 = time.time()
        charm.recordReceive(msgSize)
      arrIndex = tuple(ffi.cast(index_ctype[ndims], arrayIndex))
      charm.recvArrayMsg(aid, arrIndex, ep, ffi.buffer(msg, msgSize), t0, dcopy_start)
    except:
      charm.handleGeneralError()

  @ffi.def_extern()
  def arrayMapProcNum(gid, ndims, arrayIndex):
    try:
      arrIndex = tuple(ffi.cast(index_ctype[ndims], arrayIndex))
      return charm.arrayMapProcNum(gid, arrIndex)
    except:
      charm.handleGeneralError()

  def CkChareSend(self, chare_id, ep, msg):
    msg0, dcopy = msg
    objPtr = ffi.cast("void*", chare_id[1])
    if len(dcopy) == 0:
      lib.CkChareExtSend(chare_id[0], objPtr, ep, msg0, len(msg0))
    else:
      self.send_bufs[0] = ffi.from_buffer(msg0)
      self.send_buf_sizes[0] = len(msg0)
      for i,buf in enumerate(dcopy):
        self.send_bufs[i+1] = ffi.from_buffer(buf)
        self.send_buf_sizes[i+1] = buf.nbytes
      lib.CkChareExtSend_multi(chare_id[0], objPtr, ep, len(dcopy)+1, self.send_bufs, self.send_buf_sizes)

  def CkGroupSend(self, group_id, index, ep, msg):
    msg0, dcopy = msg
    if len(dcopy) == 0:
      lib.CkGroupExtSend(group_id, index, ep, msg0, len(msg0))
    else:
      self.send_bufs[0] = ffi.from_buffer(msg0)
      self.send_buf_sizes[0] = len(msg0)
      for i,buf in enumerate(dcopy):
        self.send_bufs[i+1] = ffi.from_buffer(buf)
        self.send_buf_sizes[i+1] = buf.nbytes
      lib.CkGroupExtSend_multi(group_id, index, ep, len(dcopy)+1, self.send_bufs, self.send_buf_sizes)

  def CkArraySend(self, array_id, index, ep, msg):
    msg0, dcopy = msg
    if len(dcopy) == 0:
      lib.CkArrayExtSend(array_id, index, len(index), ep, msg0, len(msg0))
    else:
      self.send_bufs[0] = ffi.from_buffer(msg0)
      self.send_buf_sizes[0] = len(msg0)
      for i,buf in enumerate(dcopy):
        self.send_bufs[i+1] = ffi.from_buffer(buf)
        self.send_buf_sizes[i+1] = buf.nbytes
      lib.CkArrayExtSend_multi(array_id, index, len(index), ep, len(dcopy)+1, self.send_bufs, self.send_buf_sizes)

  def CkRegisterReadonly(self, n1, n2, msg):
    if msg is None: lib.CkRegisterReadonlyExt(n1, n2, 0, ffi.NULL)
    else: lib.CkRegisterReadonlyExt(n1, n2, len(msg), msg)

  def CkRegisterMainchare(self, name, numEntryMethods):
    self.chareNames.append(ffi.new("char[]", name.encode()))
    chareIdx, startEpIdx = ffi.new("int*"), ffi.new("int*")
    lib.CkRegisterMainChareExt(self.chareNames[-1], numEntryMethods, chareIdx, startEpIdx)
    return chareIdx[0], startEpIdx[0]

  def CkRegisterGroup(self, name, numEntryMethods):
    self.chareNames.append(ffi.new("char[]", name.encode()))
    chareIdx, startEpIdx = ffi.new("int*"), ffi.new("int*")
    lib.CkRegisterGroupExt(self.chareNames[-1], numEntryMethods, chareIdx, startEpIdx)
    return chareIdx[0], startEpIdx[0]

  def CkRegisterArrayMap(self, name, numEntryMethods):
    self.chareNames.append(ffi.new("char[]", name.encode()))
    chareIdx, startEpIdx = ffi.new("int*"), ffi.new("int*")
    lib.CkRegisterArrayMapExt(self.chareNames[-1], numEntryMethods, chareIdx, startEpIdx)
    return chareIdx[0], startEpIdx[0]

  def CkRegisterArray(self, name, numEntryMethods):
    self.chareNames.append(ffi.new("char[]", name.encode()))
    chareIdx, startEpIdx = ffi.new("int*"), ffi.new("int*")
    lib.CkRegisterArrayExt(self.chareNames[-1], numEntryMethods, chareIdx, startEpIdx)
    return chareIdx[0], startEpIdx[0]

  def CkCreateGroup(self, chareIdx, epIdx, msg):
    msg0, dcopy = msg
    self.send_bufs[0] = ffi.from_buffer(msg0)
    self.send_buf_sizes[0] = len(msg0)
    for i, buf in enumerate(dcopy):
      self.send_bufs[i+1] = ffi.from_buffer(buf)
      self.send_buf_sizes[i+1] = buf.nbytes
    return lib.CkCreateGroupExt(chareIdx, epIdx, len(dcopy)+1, self.send_bufs, self.send_buf_sizes)

  def CkCreateArray(self, chareIdx, dims, epIdx, msg, map_gid):
    msg0, dcopy = msg
    ndims = len(dims)
    if all(v == 0 for v in dims): ndims = -1   # for creating an empty array Charm++ API expects ndims set to -1
    self.send_bufs[0] = ffi.from_buffer(msg0)
    self.send_buf_sizes[0] = len(msg0)
    for i, buf in enumerate(dcopy):
      self.send_bufs[i+1] = ffi.from_buffer(buf)
      self.send_buf_sizes[i+1] = buf.nbytes
    return lib.CkCreateArrayExt(chareIdx, ndims, dims, epIdx, len(dcopy)+1, self.send_bufs, self.send_buf_sizes, map_gid)

  def CkInsert(self, aid, index, epIdx, onPE, msg):
    msg0, dcopy = msg
    self.send_bufs[0] = ffi.from_buffer(msg0)
    self.send_buf_sizes[0] = len(msg0)
    for i, buf in enumerate(dcopy):
      self.send_bufs[i+1] = ffi.from_buffer(buf)
      self.send_buf_sizes[i+1] = buf.nbytes
    lib.CkInsertArrayExt(aid, len(index), index, epIdx, onPE, len(dcopy)+1, self.send_bufs, self.send_buf_sizes)

  def CkMigrate(self, aid, index, toPe):
    lib.CkMigrateExt(aid, len(index), index, toPe)

  def CkDoneInserting(self, aid):
    lib.CkArrayDoneInsertingExt(aid)

  def getTopoTreeEdges(self, pe, root_pe, pes, bfactor):
    parent       = ffi.new('int*')
    child_count  = ffi.new('int*')
    children_ptr = ffi.new("int**")
    if pes is not None:
      pes_c = ffi.new('int[]', pes)
      lib.getPETopoTreeEdges(pe, root_pe, pes_c, len(pes), bfactor, parent, child_count, children_ptr)
    else:
      lib.getPETopoTreeEdges(pe, root_pe, ffi.NULL, 0, bfactor, parent, child_count, children_ptr)
    children = [children_ptr[0][i] for i in range(child_count[0])]
    if len(children) > 0: lib.free(children_ptr[0])
    parent = parent[0]
    if parent == -1: parent = None
    return parent, children

  def start(self):
    argv_bufs = [ffi.new("char[]", arg.encode()) for arg in sys.argv]
    lib.StartCharmExt(len(sys.argv), argv_bufs)

  @ffi.def_extern()
  def arrayElemLeave(aid, ndims, arrayIndex, pdata, sizing):
    try:
      if charm.opts.PROFILING: t0 = time.time()
      if sizing:
        arrIndex = tuple(ffi.cast(index_ctype[ndims], arrayIndex))
        CharmLib.tempData = charm.arrayElemLeave(aid, arrIndex)
        pdata[0] = ffi.NULL
      else:
        pdata[0] = ffi.from_buffer(CharmLib.tempData)
      if charm.opts.PROFILING:
        global times
        times[2] += (time.time() - t0)
      return len(CharmLib.tempData)
    except:
      charm.handleGeneralError()

  @ffi.def_extern()
  def arrayElemJoin_py2(aid, ndims, arrayIndex, ep, msg, msgSize):
    try:
      t0 = None
      if charm.opts.PROFILING:
        t0 = time.time()
        charm.recordReceive(msgSize)
      arrIndex = tuple(ffi.cast(index_ctype[ndims], arrayIndex))
      charm.recvArrayMsg(aid, arrIndex, ep, ffi.buffer(msg, msgSize)[:], t0, -1)
    except:
      charm.handleGeneralError()

  @ffi.def_extern()
  def arrayElemJoin_py3(aid, ndims, arrayIndex, ep, msg, msgSize):
    try:
      t0 = None
      if charm.opts.PROFILING:
        t0 = time.time()
        charm.recordReceive(msgSize)
      arrIndex = tuple(ffi.cast(index_ctype[ndims], arrayIndex))
      charm.recvArrayMsg(aid, arrIndex, ep, ffi.buffer(msg, msgSize), t0, -1)
    except:
      charm.handleGeneralError()

  @ffi.def_extern()
  def resumeFromSync(aid, ndims, arrayIndex):
    try:
      index = tuple(ffi.cast(index_ctype[ndims], arrayIndex))
      lib.CkArrayExtSend(aid, index, ndims, charm.arrays[aid][index].thisProxy.resumeFromSync.ep,
                         emptyMsg, len(emptyMsg))
    except:
      charm.handleGeneralError()

  def CkContributeToChare(self, contributeInfo, cid):
    objPtr = ffi.cast("void*", cid[1])
    lib.CkExtContributeToChare(contributeInfo.data, cid[0], objPtr)

  def CkContributeToGroup(self, contributeInfo, gid, elemIdx):
    lib.CkExtContributeToGroup(contributeInfo.data, gid, elemIdx)

  def CkContributeToArray(self, contributeInfo, aid, index):
    lib.CkExtContributeToArray(contributeInfo.data, aid, index, len(index))

  @ffi.def_extern()
  def createReductionTargetMsg_py2(data, dataSize, reducerType, fid, returnBuffers, returnBufferSizes):
    try:
      if charm.opts.PROFILING: t0 = time.time()

      if reducerType != ReducerType.external_py:
        header = {}
        pyData = []
        if fid > 0: pyData.append(fid)
        if reducerType != ReducerType.nop:
          ctype = charm.redMgr.charm_reducer_to_ctype[reducerType]
          dataTypeTuple = c_type_table[ctype]
          numElems = dataSize // dataTypeTuple[3]
          if numElems == 1:
            pyData.append(ffi.cast(dataTypeTuple[2], data)[0])
          else:
            if haveNumpy:
              dt = charm.redMgr.rev_np_array_type_map[ctype]
              a = numpy.fromstring(ffi.buffer(data, dataSize)[:], dtype=numpy.dtype(dt))
            else:
              array_typecode = charm.redMgr.rev_array_type_map[ctype]
              a = array.array(array_typecode, ffi.buffer(data, dataSize)[:])
            pyData.append(a)

        msg = (header, pyData)
        # save msg, else it might be deleted before returning control to libcharm
        CharmLib.tempData = cPickle.dumps(msg, charm.opts.PICKLE_PROTOCOL)
        returnBuffers[0] = ffi.from_buffer(CharmLib.tempData)
        returnBufferSizes[0] = len(CharmLib.tempData)

      elif fid > 0:
        # TODO: this is INEFFICIENT. it unpickles the message, inserts the future ID
        # as first argument, and then repickles it
        # this code path is only used when the result of a reduction using a
        # Python-defined (custom) reducer is sent to a Future, which right now appears to
        # be a rare use case. But if it turns out to be critical we should consider
        # a more efficient solution
        header, args = cPickle.loads(ffi.buffer(data, dataSize)[:])
        args.insert(0, fid)
        CharmLib.tempData = cPickle.dumps((header,args), charm.opts.PICKLE_PROTOCOL)
        returnBuffers[0]     = ffi.from_buffer(CharmLib.tempData)
        returnBufferSizes[0] = len(CharmLib.tempData)
      else:
        # do nothing, use message as is (was created by charm4py)
        returnBuffers[0]     = data
        returnBufferSizes[0] = dataSize

      if charm.opts.PROFILING:
        global times
        times[0] += (time.time() - t0)

    except:
      charm.handleGeneralError()

  @ffi.def_extern()
  def createReductionTargetMsg_py3(data, dataSize, reducerType, fid, returnBuffers, returnBufferSizes):
    try:
      if charm.opts.PROFILING: t0 = time.time()

      if reducerType != ReducerType.external_py:
        header = {}
        pyData = []
        if fid > 0: pyData.append(fid)
        if reducerType != ReducerType.nop:
          ctype = charm.redMgr.charm_reducer_to_ctype[reducerType]
          dataTypeTuple = c_type_table[ctype]
          numElems = dataSize // dataTypeTuple[3]
          if numElems == 1:
            pyData.append(ffi.cast(dataTypeTuple[2], data)[0])
          else:
            if haveNumpy:
              dtype = charm.redMgr.rev_np_array_type_map[ctype]
              header[b'dcopy'] = [(len(pyData), 2, (numElems, dtype), dataSize)]
            else:
              array_typecode = charm.redMgr.rev_array_type_map[ctype]
              header[b'dcopy'] = [(len(pyData), 1, (array_typecode), dataSize)]
            returnBuffers[1] = data
            returnBufferSizes[1] = dataSize
            pyData.append(None)

        msg = (header, pyData)
        # save msg, else it might be deleted before returning control to libcharm
        CharmLib.tempData = cPickle.dumps(msg, charm.opts.PICKLE_PROTOCOL)
        returnBuffers[0] = ffi.from_buffer(CharmLib.tempData)
        returnBufferSizes[0] = len(CharmLib.tempData)

      elif fid > 0:
        # TODO: this is INEFFICIENT. it unpickles the message, inserts the future ID
        # as first argument, and then repickles it
        # this code path is only used when the result of a reduction using a
        # Python-defined (custom) reducer is sent to a Future, which right now appears to
        # be a rare use case. But if it turns out to be critical we should consider
        # a more efficient solution
        header, args = cPickle.loads(ffi.buffer(data, dataSize))
        args.insert(0, fid)
        CharmLib.tempData = cPickle.dumps((header,args), charm.opts.PICKLE_PROTOCOL)
        returnBuffers[0]     = ffi.from_buffer(CharmLib.tempData)
        returnBufferSizes[0] = len(CharmLib.tempData)
      else:
        # do nothing, use message as is (was created by charm4py)
        returnBuffers[0]     = data
        returnBufferSizes[0] = dataSize

      if charm.opts.PROFILING:
        global times
        times[0] += (time.time() - t0)

    except:
      charm.handleGeneralError()

  # callback function invoked by Charm++ for reducing contributions using a Python reducer (built-in or custom)
  @ffi.def_extern()
  def pyReduction_py2(msgs, msgSizes, nMsgs, returnBuffer):
    try:
      if charm.opts.PROFILING: t0 = time.time()
      contribs = []
      currentReducer = None
      for i in range(nMsgs):
        msgSize = msgSizes[i]
        if charm.opts.PROFILING: charm.recordReceive(msgSize)
        if msgSize > 0:
          header, args = cPickle.loads(ffi.buffer(msgs[i], msgSize)[:])
          customReducer = header[b"custom_reducer"]
          if currentReducer is None: currentReducer = customReducer
          # check for correctness of msg
          assert customReducer == currentReducer
          contribs.append(args[0])

      reductionResult = getattr(charm.reducers, currentReducer)(contribs)
      rednMsg = ({b"custom_reducer": currentReducer}, [reductionResult])
      CharmLib.tempData = cPickle.dumps(rednMsg, charm.opts.PICKLE_PROTOCOL)
      returnBuffer[0] = ffi.from_buffer(CharmLib.tempData)

      if charm.opts.PROFILING:
        global times
        times[1] += (time.time() - t0)

      return len(CharmLib.tempData)
    except:
      charm.handleGeneralError()

  # callback function invoked by Charm++ for reducing contributions using a Python reducer (built-in or custom)
  @ffi.def_extern()
  def pyReduction_py3(msgs, msgSizes, nMsgs, returnBuffer):
    try:
      if charm.opts.PROFILING: t0 = time.time()
      contribs = []
      currentReducer = None
      for i in range(nMsgs):
        msgSize = msgSizes[i]
        if charm.opts.PROFILING: charm.recordReceive(msgSize)
        if msgSize > 0:
          header, args = cPickle.loads(ffi.buffer(msgs[i], msgSize))
          customReducer = header[b"custom_reducer"]
          if currentReducer is None: currentReducer = customReducer
          # check for correctness of msg
          assert customReducer == currentReducer
          contribs.append(args[0])

      reductionResult = getattr(charm.reducers, currentReducer)(contribs)
      rednMsg = ({b"custom_reducer": currentReducer}, [reductionResult])
      CharmLib.tempData = cPickle.dumps(rednMsg, charm.opts.PICKLE_PROTOCOL)
      returnBuffer[0] = ffi.from_buffer(CharmLib.tempData)

      if charm.opts.PROFILING:
        global times
        times[1] += (time.time() - t0)

      return len(CharmLib.tempData)
    except:
      charm.handleGeneralError()

  # first callback from Charm++ shared library
  @ffi.def_extern()
  def registerMainModule():
    try:
      charm.registerMainModule()
    except:
      charm.handleGeneralError()

  def lib_version_check(self):
    commit_id = ffi.string(lib.get_charm_commit_id()).decode()
    charm.lib_version_check(commit_id)

  def init(self):

    self.lib_version_check()

    lib.registerCkRegisterMainModuleCallback(lib.registerMainModule)
    lib.registerMainchareCtorExtCallback(lib.buildMainchare)
    lib.registerArrayElemLeaveExtCallback(lib.arrayElemLeave)
    lib.registerArrayResumeFromSyncExtCallback(lib.resumeFromSync)
    lib.registerArrayMapProcNumExtCallback(lib.arrayMapProcNum)
    if sys.version_info[0] < 3:
      lib.registerReadOnlyRecvExtCallback(lib.recvReadOnly_py2)
      lib.registerChareMsgRecvExtCallback(lib.recvChareMsg_py2)
      lib.registerGroupMsgRecvExtCallback(lib.recvGroupMsg_py2)
      lib.registerArrayMsgRecvExtCallback(lib.recvArrayMsg_py2)
      lib.registerArrayElemJoinExtCallback(lib.arrayElemJoin_py2)
      lib.registerPyReductionExtCallback(lib.pyReduction_py2)
      lib.registerCreateReductionTargetMsgExtCallback(lib.createReductionTargetMsg_py2)
    else:
      lib.registerReadOnlyRecvExtCallback(lib.recvReadOnly_py3)
      lib.registerChareMsgRecvExtCallback(lib.recvChareMsg_py3)
      lib.registerGroupMsgRecvExtCallback(lib.recvGroupMsg_py3)
      lib.registerArrayMsgRecvExtCallback(lib.recvArrayMsg_py3)
      lib.registerArrayElemJoinExtCallback(lib.arrayElemJoin_py3)
      lib.registerPyReductionExtCallback(lib.pyReduction_py3)
      lib.registerCreateReductionTargetMsgExtCallback(lib.createReductionTargetMsg_py3)

    self.CkArrayExtSend = lib.CkArrayExtSend
    self.CkGroupExtSend = lib.CkGroupExtSend
    self.CkChareExtSend = lib.CkChareExtSend

    self.CkMyPe = lib.CkMyPeHook
    self.CkNumPes = lib.CkNumPesHook
    self.CkExit = lib.realCkExit
    self.CkPrintf = lib.CmiPrintf

  def CkAbort(self, msg):
    lib.CmiAbort(msg.encode())

  def LBTurnInstrumentOn(self):
    lib.LBTurnInstrumentOn()

  def LBTurnInstrumentOff(self):
    lib.LBTurnInstrumentOff()

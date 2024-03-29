# cython: language_level=3

from charm4py.charmlib.ccharm cimport *
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.stdint cimport uintptr_t
from cpython.version cimport PY_MAJOR_VERSION
from cpython.buffer  cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from cpython.tuple   cimport PyTuple_New, PyTuple_SET_ITEM
from cpython.int cimport PyInt_FromSsize_t
from cpython.ref cimport Py_INCREF

from ..charm import Charm4PyError
from .. import reduction as red
from cpython cimport array
import array

import time
import sys
if PY_MAJOR_VERSION < 3:
  from cPickle import dumps, loads
else:
  from pickle  import dumps, loads

IF HAVE_NUMPY:
  import  numpy as np
  cimport numpy as np
ELSE:
  class NumpyDummyModule:
    class ndarray: pass
    class number: pass
  np = NumpyDummyModule()

cdef object np_number = np.number


# ------ global constants ------

cdef enum:
  NUM_DCOPY_BUFS = 60   # max number of dcopy buffers

cdef enum:
  SECTION_MAX_BFACTOR = 100

cdef enum:
  MAX_INDEX_LEN = 10    # max dimensions supported for array index

# ----- reduction data structures ------

ctypedef struct CkReductionTypesExt:
  int nop
  int sum_char
  int sum_short
  int sum_int
  int sum_long
  int sum_long_long
  int sum_uchar
  int sum_ushort
  int sum_uint
  int sum_ulong
  int sum_ulong_long
  int sum_float
  int sum_double
  int product_char
  int product_short
  int product_int
  int product_long
  int product_long_long
  int product_uchar
  int product_ushort
  int product_uint
  int product_ulong
  int product_ulong_long
  int product_float
  int product_double
  int max_char
  int max_short
  int max_int
  int max_long
  int max_long_long
  int max_uchar
  int max_ushort
  int max_uint
  int max_ulong
  int max_ulong_long
  int max_float
  int max_double
  int min_char
  int min_short
  int min_int
  int min_long
  int min_long_long
  int min_uchar
  int min_ushort
  int min_uint
  int min_ulong
  int min_ulong_long
  int min_float
  int min_double
  int logical_and_bool
  int logical_or_bool
  int logical_xor_bool
  int external_py

cdef extern CkReductionTypesExt charm_reducers

class CkReductionTypesExt_Wrapper:

  def __init__(self):
    # TODO don't know if there is a way to automatically extract the fields with cython
    self.fields = {'nop': charm_reducers.nop,
              'sum_char': charm_reducers.sum_char, 'sum_short': charm_reducers.sum_short,
              'sum_int': charm_reducers.sum_int, 'sum_long': charm_reducers.sum_long,
              'sum_uchar': charm_reducers.sum_uchar, 'sum_ushort': charm_reducers.sum_ushort,
              'sum_uint': charm_reducers.sum_uint, 'sum_ulong': charm_reducers.sum_ulong,
              'sum_float': charm_reducers.sum_float, 'sum_double': charm_reducers.sum_double,
              'sum_long_long': charm_reducers.sum_long_long, 'sum_ulong_long': charm_reducers.sum_ulong_long,
              'product_char': charm_reducers.product_char, 'product_short': charm_reducers.product_short,
              'product_int': charm_reducers.product_int, 'product_long': charm_reducers.product_long,
              'product_uchar': charm_reducers.product_uchar, 'product_ushort': charm_reducers.product_ushort,
              'product_uint': charm_reducers.product_uint, 'product_ulong': charm_reducers.product_ulong,
              'product_float': charm_reducers.product_float, 'product_double': charm_reducers.product_double,
              'product_long_long': charm_reducers.product_long_long, 'product_ulong_long': charm_reducers.product_ulong_long,
              'max_char': charm_reducers.max_char, 'max_short': charm_reducers.max_short,
              'max_int': charm_reducers.max_int, 'max_long': charm_reducers.max_long,
              'max_uchar': charm_reducers.max_uchar, 'max_ushort': charm_reducers.max_ushort,
              'max_uint': charm_reducers.max_uint, 'max_ulong': charm_reducers.max_ulong,
              'max_float': charm_reducers.max_float, 'max_double': charm_reducers.max_double,
              'max_long_long': charm_reducers.max_long_long, 'max_ulong_long': charm_reducers.max_ulong_long,
              'min_char': charm_reducers.min_char, 'min_short': charm_reducers.min_short,
              'min_int': charm_reducers.min_int, 'min_long': charm_reducers.min_long,
              'min_uchar': charm_reducers.min_uchar, 'min_ushort': charm_reducers.min_ushort,
              'min_uint': charm_reducers.min_uint, 'min_ulong': charm_reducers.min_ulong,
              'min_float': charm_reducers.min_float, 'min_double': charm_reducers.min_double,
              'min_long_long': charm_reducers.min_long_long, 'min_ulong_long': charm_reducers.min_ulong_long,
              'logical_and_bool': charm_reducers.logical_and_bool,
              'logical_or_bool': charm_reducers.logical_or_bool,
              'logical_xor_bool': charm_reducers.logical_xor_bool,
              'external_py': charm_reducers.external_py}
    for f,val in self.fields.items():
      setattr(self, f, val)


ctypedef struct ContributeInfo_struct:
  int cbEpIdx            # index of entry point at reduction target
  int fid                # future ID (used when reduction target is a future)
  void *data             # data contributed for reduction
  int numelems           # number of elements in data
  int dataSize           # size of data in bytes
  int redType            # type of reduction (ReducerTypes)
  int id                 # ID of the contributing array/group
  int *idx               # index of the contributing chare array/group element
  int ndims              # number of dimensions in index
  int contributorType    # type of contributor


cdef class ContributeInfo:
  """ Each contributing chare (in group or array) has an instance of this object. """

  cdef ContributeInfo_struct internal
  cdef int index[MAX_INDEX_LEN]   # index of the chare (in C)
  # sometimes ContributeInfo needs to keep a reference to a transient buffer obj,
  # until a contribute call is completed
  cdef object buf
  cdef int bufSet # True if transient buffer has been created
  #cdef object dataSize

  def __cinit__(self, int elemId, index, int elemType):
    if isinstance(index, int): index = (index,)
    cdef int i = 0
    cdef int index_len = len(index)
    if index_len > MAX_INDEX_LEN:
      raise Charm4PyError("Element index length greater than MAX_INDEX_LEN")
    for i in range(index_len): self.index[i] = index[i]
    self.buf = None
    self.bufSet = False

    self.internal.cbEpIdx = -1
    self.internal.fid = 0
    self.internal.data = NULL
    self.internal.numelems = 0
    self.internal.dataSize = 0
    self.internal.redType = charm_reducers.nop
    self.internal.id = elemId
    self.internal.idx = self.index
    self.internal.ndims = index_len
    self.internal.contributorType = elemType

  cdef inline void setContribute(self, int ep, int fid, void *data, int numelems, int dataSize, int redType):
    #self.dataSize = dataSize
    self.internal.cbEpIdx = ep
    self.internal.fid = fid
    self.internal.data = data
    self.internal.numelems = numelems
    self.internal.dataSize = dataSize
    self.internal.redType = redType

  def getDataSize(self):
    return self.internal.dataSize

  cdef inline void setBuffer(self, buf):
    self.buf    = buf
    self.bufSet = True

  cdef inline void releaseBuffer(self):
    if self.bufSet:
      self.buf    = None
      self.bufSet = False

# -----------------------------------------

# This class is used to provide a Buffer interface to Python for messages received
# from Charm++ runtime. One of the main uses is to be able to unpickle directly
# from the message memory and avoid intermediate copies
cdef class ReceiveMsgBuffer:

  cdef Py_ssize_t shape[1]
  cdef Py_ssize_t strides[1]
  cdef char *msg

  def __cinit__(self):
    self.strides[0] = 1
    self.shape[0] = 0
    self.msg = NULL

  cdef inline int isLocal(self):
    return self.msg[0] == b'L' and self.msg[1] == b':'
  cdef inline int getLocalTag(self):
    return (<int*>(&self.msg[2]))[0]

#  def __getitem__(self, s):
#    # use this instead?:
#    #int PySlice_GetIndices(PyObject *slice, Py_ssize_t length, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step)
#    cdef Py_ssize_t start = 0
#    cdef Py_ssize_t stop  = self.shape[0]
#    if s.start is not None: start = s.start
#    if s.stop  is not None: stop  = s.stop
#    return <bytes>self.msg[start:stop:1]

  cdef inline void setMsg(self, char *_msg, int _msgSize):
    self.msg = _msg
    self.shape[0] = _msgSize

  cdef inline void setSize(self, int size):
    self.shape[0] = size

  cdef inline void advance(self, int offset):
    self.msg += offset

  def __len__(self):
    return self.shape[0]

  def __getbuffer__(self, Py_buffer *buffer, int flags):
    buffer.buf = self.msg
    buffer.format = 'b'
    buffer.internal = NULL
    buffer.itemsize = sizeof(char)
    buffer.len = self.shape[0]
    buffer.ndim = 1
    buffer.obj = self
    buffer.readonly = 1
    buffer.shape = self.shape
    buffer.strides = self.strides
    buffer.suboffsets = NULL                # for pointer arrays only

  def __releasebuffer__(self, Py_buffer *buffer):
    pass

#  cdef inline bytes tobytes(self):  # this copies msg into a bytes object
#    return <bytes>self.msg[0:self.shape[0]]


cdef inline object array_index_to_tuple(int ndims, int *arrayIndex):
  cdef int i = 0
  # TODO: not sure if there is cleaner way to make cython generate similar code
  arrIndex = PyTuple_New(ndims)
  if ndims <= 3:
    for i in range(ndims):
      d = PyInt_FromSsize_t(arrayIndex[i])
      Py_INCREF(d)
      PyTuple_SET_ITEM(arrIndex, i, d)
  else:
    for i in range(ndims):
      d = PyInt_FromSsize_t((<short*>arrayIndex)[i])
      Py_INCREF(d)
      PyTuple_SET_ITEM(arrIndex, i, d)
  return arrIndex


cdef extern const char * const CmiCommitID

# supports up to NUM_DCOPY_BUFS direct-copy entry method arguments
cdef (char*)[NUM_DCOPY_BUFS] send_bufs  # ?TODO bounds checking is needed where this is used
cdef int[NUM_DCOPY_BUFS] send_buf_sizes # ?TODO bounds checking is needed where this is used
cdef int cur_buf = 1
cdef int[MAX_INDEX_LEN] c_index
cdef Py_buffer send_buffer
cdef ReceiveMsgBuffer recv_buffer = ReceiveMsgBuffer()
cdef c_type_table_typecodes = [None] * 13
cdef int c_type_table_sizes[13]
cdef int[SECTION_MAX_BFACTOR] section_children

cdef object charm
cdef object charm_reducer_to_ctype
cdef object rev_np_array_type_map
cdef object rev_array_type_map
cdef object tempData
cdef int PROFILING = 0
cdef object PICKLE_PROTOCOL = -1
cdef object emptyMsg          # pickled empty Charm4py msg
cdef object times = [0.0] * 3 # track time in [charm reduction callbacks, custom reduction, outgoing object migration]
cdef bytes localMsg = b'L:' + (b' ' * sizeof(int))
cdef char* localMsg_ptr = <char*>localMsg


class CharmLib(object):

  def __init__(self, _charm, opts, libcharm_path):
    global charm
    charm = _charm
    self.direct_copy_supported = (PY_MAJOR_VERSION >= 3)
    self.name = 'cython'
    self.chareNames = []
    self.init()
    self.ReducerType = CkReductionTypesExt_Wrapper()
    #print(charm_reducers.sum_long, charm_reducers.product_ushort, charm_reducers.max_char, charm_reducers.max_float, charm_reducers.min_char)
    #print(ReducerType.sum_long, ReducerType.product_ushort, ReducerType.max_char, ReducerType.max_float, ReducerType.min_char)

    c_type_table_typecodes[red.C_BOOL] = 'b'
    c_type_table_typecodes[red.C_CHAR] = 'b'
    c_type_table_typecodes[red.C_SHORT] = 'h'
    c_type_table_typecodes[red.C_INT] = 'i'
    c_type_table_typecodes[red.C_LONG] = 'l'
    c_type_table_typecodes[red.C_LONG_LONG] = 'q'
    c_type_table_typecodes[red.C_UCHAR] = 'B'
    c_type_table_typecodes[red.C_USHORT] = 'H'
    c_type_table_typecodes[red.C_UINT] = 'I'
    c_type_table_typecodes[red.C_ULONG] = 'L'
    c_type_table_typecodes[red.C_ULONG_LONG] = 'Q'
    c_type_table_typecodes[red.C_FLOAT] = 'f'
    c_type_table_typecodes[red.C_DOUBLE] = 'd'

    c_type_table_sizes[red.C_BOOL] = sizeof(char)
    c_type_table_sizes[red.C_CHAR] = sizeof(char)
    c_type_table_sizes[red.C_SHORT] = sizeof(short)
    c_type_table_sizes[red.C_INT] = sizeof(int)
    c_type_table_sizes[red.C_LONG] = sizeof(long)
    c_type_table_sizes[red.C_LONG_LONG] = sizeof(long long)
    c_type_table_sizes[red.C_UCHAR] = sizeof(unsigned char)
    c_type_table_sizes[red.C_USHORT] = sizeof(unsigned short)
    c_type_table_sizes[red.C_UINT] = sizeof(unsigned int)
    c_type_table_sizes[red.C_ULONG] = sizeof(unsigned long)
    c_type_table_sizes[red.C_ULONG_LONG] = sizeof(unsigned long long)
    c_type_table_sizes[red.C_FLOAT] = sizeof(float)
    c_type_table_sizes[red.C_DOUBLE] = sizeof(double)

    self.times = times

  def sizeof(self, int c_type_id):
    return c_type_table_sizes[c_type_id]

  def getReductionTypesFields(self):
    return list(self.ReducerType.fields.keys())

  def initContributeInfo(self, elemId, index, elemType):
    return ContributeInfo(elemId, index, elemType)

  def getContributeInfo(self, int ep, int fid, tuple contribution not None, contributor not None):
    cdef ContributeInfo c_info = contributor._contributeInfo
    data = contribution[1]
    cdef int reducer_type = <int>contribution[0]
    cdef int numElems = 0
    cdef char* c_data = NULL
    cdef int c_data_size = 0
    IF HAVE_NUMPY:
      cdef np.ndarray np_array
    cdef array.array a
    if reducer_type == charm_reducers.external_py:
      numElems = <int>len(data)
      c_data = <char*>data
      c_data_size = numElems * sizeof(char)
    elif reducer_type != charm_reducers.nop:
      if isinstance(data, np.ndarray):
        np_array = data
        c_data = <char*>np_array.data
        c_data_size = np_array.nbytes # NOTE that cython's numpy C interface doesn't expose nbytes attribute
        numElems = <int>np_array.size # NOTE that cython's numpy C interface doesn't expose size attribute
      elif isinstance(data, np_number):
        PyObject_GetBuffer(data, &send_buffer, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
        try:
          c_data = <char*>send_buffer.buf
          c_data_size = <int>send_buffer.len
        finally:
          PyBuffer_Release(&send_buffer)
        numElems = 1
      elif isinstance(data, array.array):
        a = data
        numElems = <int>len(data)
        #c_data_size = a.buffer_info()[1] * a.itemsize
        c_data_size = numElems * a.itemsize # NOTE that cython's array C interface doesn't expose itemsize attribute
        c_data = <char*>a.data.as_voidptr
      else:
        c_type = contribution[2]
        # this copies and convert data to C array of C type
        a = array.array(c_type_table_typecodes[c_type], data)
        c_info.setBuffer(a)
        numElems = <int>len(data)
        c_data_size = numElems * c_type_table_sizes[c_type]
        c_data = <char*>a.data.as_voidptr

    c_info.setContribute(ep, fid, c_data, numElems, c_data_size, reducer_type)
    return c_info

  def CkChareSend(self, tuple chare_id not None, int ep, msg not None):
    global cur_buf
    msg0, dcopy = msg
    objPtr = <void*>(<uintptr_t>chare_id[1])
    if cur_buf <= 1:
      CkChareExtSend(<int>chare_id[0], objPtr, ep, msg0, len(msg0))
    else:
      send_bufs[0]      = <char*>msg0
      send_buf_sizes[0] = <int>len(msg0)
      CkChareExtSend_multi(<int>chare_id[0], objPtr, ep, cur_buf, send_bufs, send_buf_sizes)
      cur_buf = 1

  def CkGroupSend(self, int group_id, int index, int ep, msg not None):
    global cur_buf
    msg0, dcopy = msg
    if cur_buf <= 1:
      CkGroupExtSend(group_id, 1, &index, ep, msg0, len(msg0))
    else:
      send_bufs[0]      = <char*>msg0
      send_buf_sizes[0] = <int>len(msg0)
      CkGroupExtSend_multi(group_id, 1, &index, ep, cur_buf, send_bufs, send_buf_sizes)
      cur_buf = 1

  def CkGroupSendMulti(self, int group_id, list pes, int ep, msg not None):
    cdef int num_pes
    cdef int i = 0
    global cur_buf
    msg0, dcopy = msg
    num_pes = len(pes)
    assert num_pes < SECTION_MAX_BFACTOR
    for i in range(num_pes): section_children[i] = pes[i]
    if cur_buf <= 1:
      CkGroupExtSend(group_id, num_pes, section_children, ep, msg0, len(msg0))
    else:
      send_bufs[0]      = <char*>msg0
      send_buf_sizes[0] = <int>len(msg0)
      CkGroupExtSend_multi(group_id, num_pes, section_children, ep, cur_buf, send_bufs, send_buf_sizes)
      cur_buf = 1

  def CkArraySend(self, int array_id, index not None, int ep, msg not None):
    global cur_buf
    msg0, dcopy = msg
    cdef int ndims = len(index)
    cdef int i = 0
    for i in range(ndims): c_index[i] = index[i]
    if cur_buf <= 1:
      CkArrayExtSend(array_id, c_index, ndims, ep, msg0, len(msg0))
    else:
      send_bufs[0]      = <char*>msg0
      send_buf_sizes[0] = <int>len(msg0)
      CkArrayExtSend_multi(array_id, c_index, ndims, ep, cur_buf, send_bufs, send_buf_sizes)
      cur_buf = 1

  def sendToSection(self, int gid, list children):
    cdef int i = 0
    cdef int num_children
    num_children = len(children)
    assert num_children <= SECTION_MAX_BFACTOR
    for i in range(num_children):
      section_children[i] = children[i]
    CkForwardMulticastMsg(gid, num_children, section_children)

  def CkRegisterReadonly(self, bytes n1, bytes n2, msg):
    if msg is None: CkRegisterReadonlyExt(n1, n2, 0, NULL)
    else: CkRegisterReadonlyExt(n1, n2, len(msg), msg)

  def CkRegisterMainchare(self, str name, int numEntryMethods):
    self.chareNames.append(name.encode())
    cdef int chareIdx, startEpIdx
    CkRegisterMainChareExt(self.chareNames[-1], numEntryMethods, &chareIdx, &startEpIdx)
    return chareIdx, startEpIdx

  def CkRegisterGroup(self, str name, int numEntryMethods):
    self.chareNames.append(name.encode())
    cdef int chareIdx, startEpIdx
    CkRegisterGroupExt(self.chareNames[-1], numEntryMethods, &chareIdx, &startEpIdx)
    return chareIdx, startEpIdx

  def CkRegisterSectionManager(self, str name, int numEntryMethods):
    self.chareNames.append(name.encode())
    cdef int chareIdx, startEpIdx
    CkRegisterSectionManagerExt(self.chareNames[-1], numEntryMethods, &chareIdx, &startEpIdx)
    return chareIdx, startEpIdx

  def CkRegisterArrayMap(self, str name, int numEntryMethods):
    self.chareNames.append(name.encode())
    cdef int chareIdx, startEpIdx
    CkRegisterArrayMapExt(self.chareNames[-1], numEntryMethods, &chareIdx, &startEpIdx)
    return chareIdx, startEpIdx

  def CkRegisterArray(self, str name, int numEntryMethods):
    self.chareNames.append(name.encode())
    cdef int chareIdx, startEpIdx
    CkRegisterArrayExt(self.chareNames[-1], numEntryMethods, &chareIdx, &startEpIdx)
    return chareIdx, startEpIdx

  def CkCreateGroup(self, int chareIdx, int epIdx, msg not None):
    global cur_buf
    msg0, dcopy = msg
    send_bufs[0] = <char*>msg0
    send_buf_sizes[0] = <int>len(msg0)
    group_id = CkCreateGroupExt(chareIdx, epIdx, cur_buf, send_bufs, send_buf_sizes)
    cur_buf = 1
    return group_id

  def CkCreateArray(self, int chareIdx, dims not None, int epIdx, msg not None, int map_gid, char useAtSync):
    global cur_buf
    msg0, dcopy = msg
    cdef int ndims = len(dims)
    cdef int i = 0
    cdef int all_zero = 1
    for i in range(ndims):
      c_index[i] = dims[i]
      if c_index[i] != 0: all_zero = 0
    if all_zero: ndims = -1   # for creating an empty array Charm++ API expects ndims set to -1
    send_bufs[0] = <char*>msg0
    send_buf_sizes[0] = <int>len(msg0)
    array_id = CkCreateArrayExt(chareIdx, ndims, c_index, epIdx, cur_buf, send_bufs, send_buf_sizes, map_gid, useAtSync)
    cur_buf = 1
    return array_id

  def CkInsert(self, int aid, index, int epIdx, int onPE, msg not None, char useAtSync):
    global cur_buf
    msg0, dcopy = msg
    cdef int ndims = len(index)
    cdef int i = 0
    for i in range(ndims): c_index[i] = index[i]
    send_bufs[0] = <char*>msg0
    send_buf_sizes[0] = <int>len(msg0)
    CkInsertArrayExt(aid, ndims, c_index, epIdx, onPE, cur_buf, send_bufs, send_buf_sizes, useAtSync)
    cur_buf = 1

  def CkDoneInserting(self, int aid):
    CkArrayDoneInsertingExt(aid)

  def CkMigrate(self, int aid, index not None, int toPe):
    cdef int ndims = len(index)
    cdef int i = 0
    for i in range(ndims): c_index[i] = index[i]
    CkMigrateExt(aid, ndims, c_index, toPe)

  def getGroupRedNo(self, int gid):
    return CkGroupGetReductionNumber(gid)

  def getArrayElementRedNo(self, int aid, index not None):
    cdef int ndims = len(index)
    cdef int i = 0
    for i in range(ndims): c_index[i] = index[i]
    return CkArrayGetReductionNumber(aid, ndims, c_index)

  def setMigratable(self, int aid, index not None, char migratable):
    cdef int ndims = len(index)
    cdef int i = 0
    for i in range(ndims): c_index[i] = index[i]
    CkSetMigratable(aid, ndims, c_index, migratable)

  def getTopoTreeEdges(self, int pe, int root_pe, pes, int bfactor):
    cdef int parent
    cdef int child_count
    cdef int* children_ptr
    cdef int* pes_c
    cdef int num_pes
    if pes is not None:
      num_pes = len(pes)
      try:
        pes_c = <int*> malloc(num_pes * sizeof(int))
        for i in range(num_pes): pes_c[i] = pes[i]
        getPETopoTreeEdges(pe, root_pe, pes_c, num_pes, bfactor, &parent, &child_count, &children_ptr)
      finally:
        free(pes_c)
    else:
      getPETopoTreeEdges(pe, root_pe, NULL, 0, bfactor, &parent, &child_count, &children_ptr)
    children = [children_ptr[i] for i in range(child_count)]
    if child_count > 0: free(children_ptr)
    p = None
    if parent != -1: p = parent
    return p, children

  def getTopoSubtrees(self, int root_pe, list pes, int bfactor):
      cdef int parent
      cdef int child_count
      cdef int* children_ptr
      cdef int* pes_c
      cdef int num_pes
      cdef int i
      cdef int idx
      cdef int next_child = -1

      num_pes = len(pes)
      subtrees = []
      try:
          pes_c = <int*> malloc(num_pes * sizeof(int))
          for i in range(num_pes):
              pes_c[i] = pes[i]
          getPETopoTreeEdges(root_pe, root_pe, pes_c, num_pes, bfactor,
                             &parent, &child_count, &children_ptr)
          idx = 1
          for i in range(child_count):
              subtree = []
              if i < child_count - 1:
                  next_child = children_ptr[i+1]
              else:
                  next_child = -1
              while idx < num_pes:
                  pe = pes_c[idx]
                  if pe == next_child:
                      break
                  subtree.append(pe)
                  idx += 1
              subtrees.append(subtree)
          if child_count > 0:
              free(children_ptr)
      finally:
          free(pes_c)

      return subtrees

  def start(self):

    global PROFILING, PICKLE_PROTOCOL, emptyMsg
    PROFILING = <int>charm.options.profiling   # save bool in global static int variable for fast access
    PICKLE_PROTOCOL = charm.options.pickle_protocol
    emptyMsg = dumps(({},[]), PICKLE_PROTOCOL)

    global charm_reducer_to_ctype, rev_np_array_type_map, rev_array_type_map
    charm_reducer_to_ctype = charm.redMgr.charm_reducer_to_ctype
    IF HAVE_NUMPY:
      rev_np_array_type_map = charm.redMgr.rev_np_array_type_map
    rev_array_type_map = charm.redMgr.rev_array_type_map

    args = [arg.encode() for arg in sys.argv]
    cdef int num_args = len(args)
    cdef int i = 0
    # try/finally assures we don't leak memory
    try:
      argv = <char**> malloc(num_args * sizeof(char*))
      for i in range(num_args):
        argv[i] = <char*>args[i]
      StartCharmExt(num_args, argv)
    finally:
      free(argv)

  def CkContributeToChare(self, ContributeInfo contributeInfo not None, tuple cid not None):
    objPtr = <void*>(<uintptr_t>cid[1])
    CkExtContributeToChare(&contributeInfo.internal, <int>cid[0], objPtr)
    contributeInfo.releaseBuffer()

  def CkContributeToGroup(self, ContributeInfo contributeInfo not None, int gid, int elemIdx):
    CkExtContributeToGroup(&contributeInfo.internal, gid, elemIdx)
    contributeInfo.releaseBuffer()

  def CkContributeToArray(self, ContributeInfo contributeInfo not None, int aid, index not None):
    cdef int ndims = len(index)
    cdef int i = 0
    for i in range(ndims): c_index[i] = index[i]
    CkExtContributeToArray(&contributeInfo.internal, aid, c_index, ndims)
    contributeInfo.releaseBuffer()

  def CkContributeToSection(self, ContributeInfo contributeInfo not None, tuple sid, int rootPE):
    CkExtContributeToSection(&contributeInfo.internal, sid[0], sid[1], rootPE)
    contributeInfo.releaseBuffer()

  def CkStartQD_ChareCallback(self, tuple cid not None, int ep, int fid):
    objPtr = <void*>(<uintptr_t>cid[1])
    CkStartQDExt_ChareCallback(<int>cid[0], objPtr, ep, fid)

  def CkStartQD_GroupCallback(self, int gid, int pe, int ep, int fid):
    CkStartQDExt_GroupCallback(gid, pe, ep, fid)

  def CkStartQD_ArrayCallback(self, int aid, index not None, int ep, int fid):
    cdef int ndims = len(index)
    cdef int i = 0
    for i in range(ndims): c_index[i] = index[i]
    CkStartQDExt_ArrayCallback(aid, c_index, ndims, ep, fid)

  def CkStartQD_SectionCallback(self, tuple sid, int rootPE, int ep):
    CkStartQDExt_SectionCallback(sid[0], sid[1], rootPE, ep)

  def lib_version_check(self):
    charm.lib_version_check(CmiCommitID.decode('UTF-8'))

  def init(self):

    self.lib_version_check()

    registerCkRegisterMainModuleCallback(registerMainModule)
    registerMainchareCtorExtCallback(buildMainchare)
    registerArrayElemLeaveExtCallback(arrayElemLeave)
    registerArrayResumeFromSyncExtCallback(resumeFromSync)
    registerReadOnlyRecvExtCallback(recvReadOnly)
    registerChareMsgRecvExtCallback(recvChareMsg)
    registerGroupMsgRecvExtCallback(recvGroupMsg)
    registerArrayMsgRecvExtCallback(recvArrayMsg)
    registerArrayBcastRecvExtCallback(recvArrayBcast)
    registerArrayMapProcNumExtCallback(arrayMapProcNum)
    registerArrayElemJoinExtCallback(arrayElemJoin)
    registerPyReductionExtCallback(pyReduction)
    registerCreateCallbackMsgExtCallback(createCallbackMsg)

  def CkMyPe(self): return CkMyPeHook()
  def CkNumPes(self): return CkNumPesHook()
  def CkExit(self, int exitCode): return realCkExit(exitCode)
  def CkPrintf(self, bytes msg): CmiPrintf("%s", msg)
  def CkAbort(self, str msg): return CmiAbort("%s", <bytes>msg.encode())
  def LBTurnInstrumentOn(self):  LBTurnInstrumentOn()
  def LBTurnInstrumentOff(self): LBTurnInstrumentOff()
  def CkGetFirstPeOnPhysicalNode(self, int node): return CmiGetFirstPeOnPhysicalNode(node)
  def CkPhysicalNodeID(self, int pe): return CmiPhysicalNodeID(pe)
  def CkNumPhysicalNodes(self): return CmiNumPhysicalNodes()
  def CkNumPesOnPhysicalNode(self, int node): return CmiNumPesOnPhysicalNode(node)
  def CkPhysicalRank(self, int pe): return CmiPhysicalRank(pe)

  def CkGetPesOnPhysicalNode(self, int node):
    cdef int numpes
    cdef int *pelist
    cdef int i = 0
    CmiGetPesOnPhysicalNode(node, &pelist, &numpes)
    return [pelist[i] for i in range(numpes)]

  def unpackMsg(self, ReceiveMsgBuffer msg not None, int dcopy_start, dest_obj):
    cdef int i = 0
    cdef int buf_size
    cdef int typeId
    if msg.isLocal():
      header, args = dest_obj.__removeLocal__(msg.getLocalTag())
    else:
      header, args = loads(msg)
      if b'dcopy' in header:
        msg.advance(dcopy_start)
        dcopy_list = header[b'dcopy']
        for i in range(len(dcopy_list)):
          arg_pos, tid, rebuildArgs, size = dcopy_list[i]
          typeId = <int>tid
          buf_size = <int>size
          msg.setSize(buf_size)
          if typeId == 0:
            args[arg_pos] = bytes(msg)
          elif typeId == 1:
            typecode = rebuildArgs[0]
            a = array.array(typecode)
            a.frombytes(msg)
            args[arg_pos] = a
          elif typeId == 2:
            shape, dt = rebuildArgs
            a = np.frombuffer(msg, dtype=np.dtype(dt))  # this does not copy
            a.shape = shape
            args[arg_pos] = a.copy()
          else:
            raise Charm4PyError("unpackMsg: wrong type id received")
          msg.advance(buf_size)
      elif b"custom_reducer" in header:
        reducer = getattr(charm.reducers, header[b"custom_reducer"])
        # reduction result won't always be in position 0, but will always be last
        # (e.g. if reduction target is a future, the reduction result will be 2nd argument)
        if reducer.hasPostprocess: args[-1] = reducer.postprocess(args[-1])

    return header, args

  def packMsg(self, destObj, msgArgs not None, dict header):
    cdef int i = 0
    cdef int localTag
    cdef array.array a
    IF HAVE_NUMPY:
      cdef np.ndarray np_array
    dcopy_size = 0
    if destObj is not None: # if dest obj is local
      localTag = destObj.__addLocal__((header, msgArgs))
      memcpy(localMsg_ptr+2, &localTag, sizeof(int))
      msg = localMsg
    elif len(msgArgs) == 0 and len(header) == 0:
      msg = emptyMsg
    else:
      direct_copy_hdr = []  # goes to header
      args = list(msgArgs)
      global cur_buf
      cur_buf = 1
      for i in range(len(args)):
        arg = msgArgs[i]
        if isinstance(arg, np.ndarray) and not arg.dtype.hasobject:
          np_array = arg
          nbytes = np_array.nbytes
          if arg.dtype.isbuiltin:
            direct_copy_hdr.append((i, 2, (arg.shape, arg.dtype.char), nbytes))
          else:
            direct_copy_hdr.append((i, 2, (arg.shape, arg.dtype.name), nbytes))
          send_bufs[cur_buf] = <char*>np_array.data
        elif isinstance(arg, bytes):
          nbytes = len(arg)
          direct_copy_hdr.append((i, 0, (), nbytes))
          send_bufs[cur_buf] = <char*>arg
        elif isinstance(arg, array.array):
          a = arg
          #nbytes = arg.buffer_info()[1] * arg.itemsize
          nbytes = len(a) * a.itemsize # NOTE that cython's array C interface doesn't expose itemsize attribute
          direct_copy_hdr.append((i, 1, (a.typecode), nbytes))
          send_bufs[cur_buf] = <char*>a.data.as_voidptr
        else:
          continue
        args[i] = None  # will direct-copy this arg so remove from args list
        send_buf_sizes[cur_buf] = <int>nbytes
        if PROFILING: dcopy_size += nbytes
        cur_buf += 1
      if len(direct_copy_hdr) > 0: header[b'dcopy'] = direct_copy_hdr
      try:
        msg = dumps((header, args), PICKLE_PROTOCOL)
      except:
        global cur_buf
        cur_buf = 1
        raise
    if PROFILING: charm.recordSend(len(msg) + dcopy_size)
    return msg, None

  def scheduleTagAfter(self, int tag, double msecs):
    CcdCallFnAfter(CcdCallFnAfterCallback, <void*>tag, msecs)


# first callback from Charm++ shared library
cdef void registerMainModule() noexcept:
  try:
    charm.registerMainModule()
  except:
    charm.handleGeneralError()

cdef void recvReadOnly(int msgSize, char *msg) noexcept:
  try:
    recv_buffer.setMsg(msg, msgSize)
    charm.recvReadOnly(recv_buffer)
  except:
    charm.handleGeneralError()

cdef void buildMainchare(int onPe, void *objPtr, int ep, int argc, char **argv) noexcept:
  try:
    args = [argv[i].decode('UTF-8') for i in range(argc)]
    charm.buildMainchare(onPe, <uintptr_t> objPtr, ep, args)
  except:
    charm.handleGeneralError()

cdef void recvChareMsg(int onPe, void *objPtr, int ep, int msgSize, char *msg, int dcopy_start) noexcept:
  try:
    if PROFILING:
      charm._precvtime = time.time()
      charm.recordReceive(msgSize)
    recv_buffer.setMsg(msg, msgSize)
    charm.recvChareMsg((onPe, <uintptr_t>objPtr), ep, recv_buffer, dcopy_start)
  except:
    charm.handleGeneralError()

cdef void recvGroupMsg(int gid, int ep, int msgSize, char *msg, int dcopy_start) noexcept:
  try:
    if PROFILING:
      charm._precvtime = time.time()
      charm.recordReceive(msgSize)
    recv_buffer.setMsg(msg, msgSize)
    charm.recvGroupMsg(gid, ep, recv_buffer, dcopy_start)
  except:
    charm.handleGeneralError()

cdef void recvArrayMsg(int aid, int ndims, int *arrayIndex, int ep, int msgSize, char *msg, int dcopy_start) noexcept:
  try:
    if PROFILING:
      charm._precvtime = time.time()
      charm.recordReceive(msgSize)
    recv_buffer.setMsg(msg, msgSize)
    charm.recvArrayMsg(aid, array_index_to_tuple(ndims, arrayIndex), ep, recv_buffer, dcopy_start)
  except:
    charm.handleGeneralError()

cdef void recvArrayBcast(int aid, int ndims, int nInts, int numElems, int *arrayIndexes, int ep, int msgSize, char *msg, int dcopy_start) noexcept:
  cdef int i = 0
  try:
    if PROFILING:
      charm._precvtime = time.time()
      charm.recordReceive(msgSize)
    recv_buffer.setMsg(msg, msgSize)
    indexes = []
    for i in range(numElems):
        indexes.append(array_index_to_tuple(ndims, arrayIndexes))
        arrayIndexes += nInts
    charm.recvArrayBcast(aid, indexes, ep, recv_buffer, dcopy_start)
  except:
    charm.handleGeneralError()

cdef int arrayMapProcNum(int gid, int ndims, const int *arrayIndex) noexcept:
  try:
    return charm.arrayMapProcNum(gid, array_index_to_tuple(ndims, arrayIndex))
  except:
    charm.handleGeneralError()

cdef int arrayElemLeave(int aid, int ndims, int *arrayIndex, char **pdata, int sizing) noexcept:
  cdef int i = 0
  global tempData
  try:
    if PROFILING: t0 = time.time()
    if sizing:
      tempData = charm.arrayElemLeave(aid, array_index_to_tuple(ndims, arrayIndex))
      pdata[0] = NULL
    else:
      pdata[0] = <char*>tempData
    if PROFILING:
      times[2] += (time.time() - t0)
    return len(tempData)
  except:
    charm.handleGeneralError()

cdef void arrayElemJoin(int aid, int ndims, int *arrayIndex, int ep, char *msg, int msgSize) noexcept:
  cdef int i = 0
  try:
    if PROFILING:
      charm._precvtime = time.time()
      charm.recordReceive(msgSize)
    recv_buffer.setMsg(msg, msgSize)
    charm.recvArrayMsg(aid, array_index_to_tuple(ndims, arrayIndex), ep, recv_buffer, -1)
  except:
    charm.handleGeneralError()

cdef void resumeFromSync(int aid, int ndims, int *arrayIndex) noexcept:
  cdef int i = 0
  try:
    index = array_index_to_tuple(ndims, arrayIndex)
    CkArrayExtSend(aid, arrayIndex, ndims, charm.arrays[aid][index].thisProxy.resumeFromSync.ep,
                       emptyMsg, len(emptyMsg))
  except:
    charm.handleGeneralError()

cdef void createCallbackMsg(void *data, int dataSize, int reducerType, int fid, int *sectionInfo,
                            char **returnBuffers, int *returnBufferSizes) noexcept:
  cdef int numElems
  cdef array.array a
  cdef int item_size
  global tempData
  try:
    if PROFILING: t0 = time.time()

    pyData = []
    if sectionInfo[0] >= 0:
      # this is a section callback
      sid = (sectionInfo[0], sectionInfo[1])
      pyData = [sid, sectionInfo[2], {b'sid': sid}]
      secMgrProxy = charm.sectionMgr.thisProxy
      # tell Charm++ the gid and ep of SectionManager for section broadcasts
      sectionInfo[0] = <int>secMgrProxy.gid
      sectionInfo[1] = <int>secMgrProxy.sendToSection.ep

    if (reducerType < 0) or (reducerType == charm_reducers.nop):
      if fid > 0:
        tempData = dumps(({}, [fid]), PICKLE_PROTOCOL)
      elif len(pyData) == 0:
        tempData = emptyMsg
      else:
        # section
        tempData = dumps(({}, pyData), PICKLE_PROTOCOL)
      returnBuffers[0]     = <char*>tempData
      returnBufferSizes[0] = len(tempData)

    elif reducerType != charm_reducers.external_py:

      header = {}
      ctype = charm_reducer_to_ctype[reducerType]
      item_size = c_type_table_sizes[ctype]
      numElems = dataSize // item_size # force integer division for cython + python3
      if fid > 0:
        pyData.append(fid)
      if numElems == 1:
        a = array.array(c_type_table_typecodes[ctype], [0])
        memcpy(a.data.as_voidptr, data, item_size)
        pyData.append(a[0])
      else:
        IF HAVE_NUMPY:
          dtype = rev_np_array_type_map[ctype]
          header[b'dcopy'] = [(len(pyData), 2, (numElems, dtype), dataSize)]
        ELSE:
          array_typecode = rev_array_type_map[ctype]
          header[b'dcopy'] = [(len(pyData), 1, (array_typecode), dataSize)]
        returnBuffers[1]     = <char*>data
        returnBufferSizes[1] = dataSize
        pyData.append(None)
      # save msg, else it might be deleted before returning control to libcharm
      tempData = dumps((header, pyData), PICKLE_PROTOCOL)
      returnBuffers[0]     = <char*>tempData
      returnBufferSizes[0] = len(tempData)

    elif fid > 0 or len(pyData) > 0:
      # TODO: this is INEFFICIENT. it unpickles the message, then either:
      # a) inserts the future ID as first argument
      # b) puts the data into a section msg
      # then repickles the message.
      # this code path is only used when the result of a reduction using a
      # Python-defined (custom) reducer is sent to a Future or Section.
      # If this turns out to be critical we should consider a more efficient solution
      recv_buffer.setMsg(<char*>data, dataSize)
      header, args = loads(recv_buffer)
      if fid > 0:
        args.insert(0, fid)
      else:
        pyData.extend(args)
        args = pyData
      tempData = dumps((header, args), PICKLE_PROTOCOL)
      returnBuffers[0]     = <char*>tempData
      returnBufferSizes[0] = len(tempData)
    else:
      # do nothing, use message as is (was created by Charm4py)
      returnBuffers[0]     = <char*>data
      returnBufferSizes[0] = dataSize

    if PROFILING:
      times[0] += (time.time() - t0)

  except:
    charm.handleGeneralError()

# callback function invoked by Charm++ for reducing contributions using a Python reducer (built-in or custom)
cdef int pyReduction(char** msgs, int* msgSizes, int nMsgs, char** returnBuffer) noexcept:
  cdef int i = 0
  cdef int msgSize
  global tempData
  try:
    if PROFILING: t0 = time.time()
    contribs = []
    currentReducer = None
    for i in range(nMsgs):
      msgSize = msgSizes[i]
      if PROFILING: charm.recordReceive(msgSize)
      if msgSize > 0:
        recv_buffer.setMsg(msgs[i], msgSize)
        header, args = loads(recv_buffer)
        customReducer = header[b'custom_reducer']
        if currentReducer is None: currentReducer = customReducer
        # check for correctness of msg
        assert customReducer == currentReducer
        contribs.append(args[0])

    # apply custom reducer
    reductionResult = getattr(charm.reducers, currentReducer)(contribs)
    rednMsg = ({b"custom_reducer": currentReducer}, [reductionResult])
    tempData = dumps(rednMsg, PICKLE_PROTOCOL)
    returnBuffer[0] = <char*>tempData

    if PROFILING:
      times[1] += (time.time() - t0)

    return len(tempData)
  except:
    charm.handleGeneralError()

cdef void CcdCallFnAfterCallback(void *userParam, double curWallTime) noexcept:
  try:
    charm.triggerCallable(<int>userParam)
  except:
    charm.handleGeneralError()

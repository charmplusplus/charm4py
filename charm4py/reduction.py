import array
from functools import reduce
import operator as op
from itertools import chain
import sys
try:
    import numpy as np
    haveNumpy = True
except ImportError:
    # this is to avoid numpy dependency
    haveNumpy = False
    class NumpyDummyModule:
        class ndarray:
            pass
        class number:
            pass
    np = NumpyDummyModule()


from . import _ckdyn

# identifiers for Charm internal reducers
(SUM, PRODUCT, MAX, MIN, AND, OR, XOR) = range(7)

# CKDYN_ELEM_* element kinds (ckdyn.h)
(ELEM_NONE, ELEM_I8, ELEM_I16, ELEM_I32, ELEM_I64,
 ELEM_U8, ELEM_U16, ELEM_U32, ELEM_U64,
 ELEM_F16, ELEM_F32, ELEM_F64, ELEM_C64, ELEM_C128,
 ELEM_BOOL, ELEM_BYTE) = range(16)
NUM_ELEMS = 16

# numpy dtype name of each element kind that numpy and the built-in reducers share
elem_to_numpy = {ELEM_I8: 'int8', ELEM_I16: 'int16', ELEM_I32: 'int32', ELEM_I64: 'int64',
                 ELEM_U8: 'uint8', ELEM_U16: 'uint16', ELEM_U32: 'uint32', ELEM_U64: 'uint64',
                 ELEM_F16: 'float16', ELEM_F32: 'float32', ELEM_F64: 'float64',
                 ELEM_C64: 'complex64', ELEM_C128: 'complex128', ELEM_BOOL: 'bool'}

# array.array typecode of each element kind (used when numpy is absent)
elem_to_typecode = {ELEM_I8: 'b', ELEM_I16: 'h', ELEM_I32: 'i', ELEM_I64: 'q',
                    ELEM_U8: 'B', ELEM_U16: 'H', ELEM_U32: 'I', ELEM_U64: 'Q',
                    ELEM_F32: 'f', ELEM_F64: 'd', ELEM_BOOL: 'b'}

# reducer name prefix in CkDynReducers() -> op
_op_names = {'sum': SUM, 'product': PRODUCT, 'max': MAX, 'min': MIN,
             'logical_and': AND, 'logical_or': OR, 'logical_xor': XOR}


def _useNumpyForReduction(contribs):
    isNumpyType = type(contribs) == np.ndarray or type(contribs[0]) == np.ndarray
    # always prefer numpy when we can use it to take advantage of speed
    # also, the non-section version will return numpy arrays when possible
    return haveNumpy or isNumpyType
# ------------------- Reducers -------------------


def _elementwiseOp(op, data):
    return reduce(op, data)


# apply an op to pairwise elements in a list of lists
def _pairwiseOp(op, data):
    result = data[0]
    for i in range(1, len(data)):
        for j in range(len(data[i])):
            result[j] = op(result[j], data[i][j])
    return result


# python versions of built-in reducers
def _sum(contribs):
    if _useNumpyForReduction(contribs):
        return np.add.reduce(contribs)

    try:
        return sum(contribs)
    except TypeError:
        return _pairwiseOp(op.add, contribs)


def _product(contribs):
    if _useNumpyForReduction(contribs):
        return np.multiply.reduce(contribs)

    try:
        return _elementwiseOp(op.mul, contribs)
    except TypeError:
        return _pairwiseOp(op.mul, contribs)


def _max(contribs):
    if _useNumpyForReduction(contribs):
        return np.maximum.reduce(contribs)

    try:
        return max(contribs)
    except TypeError:
        return _pairwiseOp(max, contribs)


def _min(contribs):
    if _useNumpyForReduction(contribs):
        return np.minimum.reduce(contribs)

    try:
        return min(contribs)
    except TypeError:
        return _pairwiseOp(min, contribs)


def _and(contribs):
    if _useNumpyForReduction(contribs):
        return np.logical_and.reduce(contribs)

    try:
        iter(contribs[0])
        return _pairwiseOp(lambda x, y: x and y, contribs)
    except TypeError:
        return _elementwiseOp(lambda x, y: x and y, contribs)


def _or(contribs):
    if _useNumpyForReduction(contribs):
        return np.logical_or.reduce(contribs)

    try:
        iter(contribs[0])
        return _pairwiseOp(lambda x, y: x or y, contribs)
    except TypeError:
        return _elementwiseOp(lambda x, y: x or y, contribs)


def _xor(contribs):
    if _useNumpyForReduction(contribs):
        return np.logical_xor.reduce(contribs)

    try:
        iter(contribs[0])
        return _pairwiseOp(lambda x, y: bool(x) ^ bool(y), contribs)
    except TypeError:
        return _elementwiseOp(lambda x, y: bool(x) ^ bool(y), contribs)


def _bcast_exc_reducer(contribs):
    # return first non empty contribution
    for c in contribs:
        if c is not None:
            return c


def gather(contribs):
    # contribs will be a list of list of tuples
    # first element of tuple is always array index of chare
    return sorted(chain(*contribs))


def gather_preprocess(data, contributor):
    return [(contributor.thisIndex, data)]


def gather_postprocess(contrib):
    return [tup[1] for tup in contrib]


class _NopReducer(object):
    """ the nop reducer (a sentinel) """
    def __repr__(self):
        return 'Reducer.nop'


class ReducerContainer(object):

    def __init__(self, charm):
        self._by_id = []  # reducer id -> function, in registration order
        self.addReducer(_sum)
        self.addReducer(_product)
        self.addReducer(_max)
        self.addReducer(_min)
        self.addReducer(_and)
        self.addReducer(_or)
        self.addReducer(_xor)
        self.addReducer(_bcast_exc_reducer)
        self.addReducer(gather, pre=gather_preprocess, post=gather_postprocess)

        self.nop     = _NopReducer()
        self.sum     = (SUM,     self._sum)     # (internal op code, python reducer)
        self.product = (PRODUCT, self._product)
        self.max     = (MAX,     self._max)
        self.min     = (MIN,     self._min)
        self.logical_and = (AND, self._and)
        self.logical_or  = (OR,  self._or)
        self.logical_xor = (XOR, self._xor)

    def addReducer(self, func, pre=None, post=None):
        if hasattr(self, func.__name__):
            from .charm import Charm4PyError
            raise Charm4PyError("Reducer with name " + func.__name__ + " already registered")
        func.hasPreprocess  = False
        func.hasPostprocess = False
        if pre is not None:
            func.hasPreprocess = True
            func.preprocess = pre
        if post is not None:
            func.hasPostprocess = True
            func.postprocess = post
        # the id travels in the contribution; every process registers its
        # reducers in the same order, so ids agree
        func._rid = len(self._by_id)
        self._by_id.append(func)
        setattr(self, func.__name__, func)


# ------------------- Reduction Manager -------------------

class ReductionManager(object):

    def __init__(self, charm, reducers):
        self.charm = charm
        self.reducers = reducers
        self._foreign_type = None
        self.populateConversionTables()

    @property
    def reducers_by_id(self):
        return self.reducers._by_id

    @property
    def foreign_type(self):
        # CkDyn registers the foreign reducer when the runtime starts
        if self._foreign_type is None:
            self._foreign_type = _ckdyn.foreign_reducer_type()
        return self._foreign_type

    def populateConversionTables(self):
        # red_table[op][elem] is the Charm++ reducer type for the built-in op
        # on elements of kind elem, or None
        self.red_table = [[None] * NUM_ELEMS for _ in range(7)]
        self.nop_type = None
        for name, rtype, elem in _ckdyn.reducers():
            if name == 'nop':
                self.nop_type = rtype
                continue
            op = None
            for prefix, op_id in _op_names.items():
                if name == prefix or name.startswith(prefix + '_'):
                    op = op_id
                    break
            if op is None:
                continue
            if op in (AND, OR, XOR) and name not in (prefix, prefix + '_int', prefix + '_bool'):
                continue
            if self.red_table[op][elem] is None:  # when two names share an elem, the first wins
                self.red_table[op][elem] = rtype
        assert self.nop_type is not None, 'CkDyn has no nop reducer'

        # ------ numpy data types ------
        if haveNumpy:
            # map numpy data types to element kinds
            self.numpy_type_map = {'bool': ELEM_BOOL, 'int8': ELEM_I8, 'int16': ELEM_I16,
                                   'int32': ELEM_I32, 'int64': ELEM_I64, 'uint8': ELEM_U8,
                                   'uint16': ELEM_U16, 'uint32': ELEM_U32, 'uint64': ELEM_U64,
                                   'float16': ELEM_F16, 'float32': ELEM_F32, 'float64': ELEM_F64,
                                   'complex64': ELEM_C64, 'complex128': ELEM_C128}

        # ------ array.array data types ------
        self.array_type_map = {}
        for tc, elem in (('b', ELEM_I8), ('B', ELEM_U8), ('h', ELEM_I16), ('H', ELEM_U16),
                         ('f', ELEM_F32), ('d', ELEM_F64)):
            self.array_type_map[tc] = elem
        for tc in ('i', 'l', 'q'):
            self.array_type_map[tc] = {1: ELEM_I8, 2: ELEM_I16, 4: ELEM_I32, 8: ELEM_I64}[array.array(tc).itemsize]
        for tc in ('I', 'L', 'Q'):
            self.array_type_map[tc] = {1: ELEM_U8, 2: ELEM_U16, 4: ELEM_U32, 8: ELEM_U64}[array.array(tc).itemsize]

        # ------ python data types ------
        self.python_type_map = {int: ELEM_I64, float: ELEM_F64, bool: ELEM_BOOL}
        if haveNumpy:
            # this is a bit of a hack
            self.python_type_map[np.bool_] = ELEM_BOOL

    def _to_buffer(self, data, elem):
        """ a Python scalar or a list of them as a contiguous buffer of elem """
        if not isinstance(data, (list, tuple)):
            data = [data]
        if haveNumpy:
            return np.array(data, dtype=elem_to_numpy[elem])
        return array.array(elem_to_typecode[elem], data)

    # return (Charm++ reducer type, data ready for _ckdyn.contribute)
    def prepare(self, data, reducer, contributor):
        if (reducer is None) or (reducer is self.reducers.nop):
            return (self.nop_type, None)

        rtype = None
        if type(reducer) == tuple:
            op, py_red_func = reducer
            pyReducer = None
            dt = type(data)
            if isinstance(data, np.ndarray) or isinstance(data, np.number):
                if not data.dtype.hasobject and data.dtype.name in self.numpy_type_map:
                    rtype = self.red_table[op][self.numpy_type_map[data.dtype.name]]
                    data = np.ascontiguousarray(data)
                else:
                    pyReducer = py_red_func
            elif dt == array.array:
                rtype = self.red_table[op][self.array_type_map[data.typecode]]
            else:
                try:
                    elem = self.python_type_map[dt]
                    rtype = self.red_table[op][elem]
                    data = self._to_buffer(data, elem)
                except:
                    try:
                        t0 = type(data[0])
                        elem = self.python_type_map[t0]
                        rtype = self.red_table[op][elem]
                        for x in data:
                            if type(x) != t0:
                                pyReducer = py_red_func
                                break
                        if pyReducer is None:
                            data = self._to_buffer(data, elem)
                    except:
                        pyReducer = py_red_func
            if pyReducer is not None:
                # A built-in reducer on data with no Charm++ type (a torch tensor, an
                # object with __array__, a list of mixed types): the Python reducer
                # combines the contributions, but Charm++ does not call a reducer when
                # a reduction has one contributor, so that result would be the object
                # as contributed. Convert the contribution the way the reducer does
                # (the built-in Python reducers return numpy arrays), so that the
                # result type does not depend on the number of contributors; a numeric
                # array then uses the Charm++ reducer.
                try:
                    conv = py_red_func([data])
                except Exception:
                    conv = data
                if (isinstance(conv, np.ndarray) and not conv.dtype.hasobject
                        and conv.dtype.name in self.numpy_type_map):
                    rtype = self.red_table[op][self.numpy_type_map[conv.dtype.name]]
                    data = np.ascontiguousarray(conv)
                    pyReducer = None
                else:
                    data = conv
        else:
            pyReducer = reducer

        if pyReducer is None:
            assert rtype is not None, 'Could not find a valid reducer. Check that datatype matches the operator'
            return (rtype, data)
        else:
            if not hasattr(pyReducer, '_rid'):
                from .charm import Charm4PyError
                raise Charm4PyError('Invalid reducer ' + str(reducer) + '. Reducers must be functions registered with addReducer')
            if pyReducer.hasPreprocess:
                data = pyReducer.preprocess(data, contributor)
            return (self.foreign_type, (pyReducer._rid, data))

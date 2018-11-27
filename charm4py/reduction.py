import sys
if sys.version_info[0] < 3:
    import cPickle
else:
    import pickle as cPickle
from itertools import chain
import array
try:
    import numpy as np
    haveNumpy = True
except ImportError:
    # this is to avoid numpy dependency
    haveNumpy = False
    class NumpyDummyModule:
        class ndarray: pass
        class number: pass
    np = NumpyDummyModule()


# identifiers for Charm internal reducers
(SUM, PRODUCT, MAX, MIN) = range(4)

NUM_C_TYPES = 12
# Set of integer identifiers for C types used with internal reducers
(C_CHAR,  C_SHORT,  C_INT,  C_LONG,  C_LONG_LONG,
 C_UCHAR, C_USHORT, C_UINT, C_ULONG, C_ULONG_LONG,
 C_FLOAT, C_DOUBLE) = range(NUM_C_TYPES)

# map names of C types (as they appear in CkReductionTypesExt) to their identifiers
c_typename_to_id = {'char':  C_CHAR,  'short':  C_SHORT,  'int':  C_INT,  'long':  C_LONG,  'long_long':  C_LONG_LONG,
                    'uchar': C_UCHAR, 'ushort': C_USHORT, 'uint': C_UINT, 'ulong': C_ULONG, 'ulong_long': C_ULONG_LONG,
                    'float': C_FLOAT, 'double': C_DOUBLE}


# ------------------- Reducers -------------------

# python versions of built-in reducers
def _sum(contribs):
    return sum(contribs)

def _product(contribs):
    result = contribs[0]
    for i in range(1, len(contribs)):
        result *= contribs[i]
    return result

def _max(contribs):
    return max(contribs)

def _min(contribs):
    return min(contribs)

def gather(contribs):
    # contribs will be a list of list of tuples
    # first element of tuple is always array index of chare
    return sorted(chain(*contribs))

def gather_preprocess(data, contributor):
    return [(contributor.thisIndex, data)]

def gather_postprocess(contrib):
    return [tup[1] for tup in contrib]


class ReducerContainer(object):

    def __init__(self, charm):
        self.addReducer(_sum)
        self.addReducer(_product)
        self.addReducer(_max)
        self.addReducer(_min)
        self.addReducer(gather, pre=gather_preprocess, post=gather_postprocess)

        self.nop     = charm.ReducerType.nop
        self.sum     = (SUM,     self._sum)     # (internal op code, python reducer)
        self.product = (PRODUCT, self._product)
        self.max     = (MAX,     self._max)
        self.min     = (MIN,     self._min)

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
        setattr(self, func.__name__, func)


# ------------------- Reduction Manager -------------------

class ReductionManager(object):

    def __init__(self, charm, reducers):
        self.charm = charm
        self.reducers = reducers
        self.populateConversionTables()

    def populateConversionTables(self):

        import os

        # `red_table[op][c_type]` maps to `charm_reducer_type`, where:
        #     - op is the identifier for internal reducer (SUM, PRODUCT, MAX or INT)
        #     - c_type is identifier for C type (C_CHAR, C_SHORT, etc)
        #     - charm_reducer_type is value for internal reducer type as they appear in CkReductionTypesExt
        self.red_table = [[]] * 4
        self.red_table[SUM]     = [0] * NUM_C_TYPES
        self.red_table[PRODUCT] = [0] * NUM_C_TYPES
        self.red_table[MAX]     = [0] * NUM_C_TYPES
        self.red_table[MIN]     = [0] * NUM_C_TYPES

        fields = self.charm.lib.getReductionTypesFields()  # get names of fields in CkReductionTypesExt
        maxFieldVal = max([getattr(self.charm.ReducerType, f) for f in fields])
        # charm_reducer_to_ctype maps the values in CkReductionTypesExt to C type identifier
        self.charm_reducer_to_ctype = [None] * (maxFieldVal + 1)
        for f in fields:
            if f == 'nop':
                continue
            elif f == 'external_py':
                op, c_type_str = None, 'char'
            else:
                op, c_type_str = f.split('_', 1)        # e.g. from 'sum_long' extracts 'sum' and 'long'
            ctype_code = c_typename_to_id[c_type_str]   # e.g. map 'long' to C_LONG
            f_val = getattr(self.charm.ReducerType, f)  # value of the field in CkReductionTypesExt
            # print(f, "ctype_code", ctype_code, "f_val=", f_val)
            self.charm_reducer_to_ctype[f_val] = ctype_code
            if   op == 'sum':     self.red_table[SUM][ctype_code] = f_val
            elif op == 'product': self.red_table[PRODUCT][ctype_code] = f_val
            elif op == 'max':     self.red_table[MAX][ctype_code] = f_val
            elif op == 'min':     self.red_table[MIN][ctype_code] = f_val

        # ------ numpy data types ------
        if haveNumpy:

            # map numpy data types to internal reduction C code identifier
            self.numpy_type_map = {'bool': C_CHAR, 'int8': C_CHAR, 'int16': C_SHORT,
                                   'int32': C_INT, 'int64': C_LONG, 'uint8': C_UCHAR,
                                   'uint16': C_USHORT, 'uint32': C_UINT, 'uint64': C_ULONG,
                                   #'float16': ?
                                   'float32': C_FLOAT, 'float64': C_DOUBLE}
            if os.name == 'nt':
                self.numpy_type_map['int64']  = C_LONG_LONG
                self.numpy_type_map['uint64'] = C_ULONG_LONG

            # verify that mapping is correct
            for dt, c_type in self.numpy_type_map.items():
                assert np.dtype(dt).itemsize == self.charm.lib.sizeof(c_type)

            self.rev_np_array_type_map = [None] * NUM_C_TYPES
            reverse_lookup = {v: k for k, v in self.numpy_type_map.items()}
            for c_type in range(NUM_C_TYPES):
                if c_type in reverse_lookup:
                    self.rev_np_array_type_map[c_type] = reverse_lookup[c_type]

        # ------ array.array data types ------

        # map array.array data types to internal reduction C code identifier
        self.array_type_map = {'b': C_CHAR, 'B': C_UCHAR, 'h': C_SHORT, 'H': C_USHORT,
                               'i': C_INT, 'I': C_UINT, 'l': C_LONG, 'L': C_ULONG,
                               'f': C_FLOAT, 'd': C_DOUBLE}
        if sys.version_info >= (3, 3, 0):
            self.array_type_map['q'] = C_LONG_LONG
            self.array_type_map['Q'] = C_ULONG_LONG

        # verify that mapping is correct
        for dt, c_type in self.array_type_map.items():
            assert array.array(dt).itemsize == self.charm.lib.sizeof(c_type)

        self.rev_array_type_map = ['b', 'h', 'i', 'l', 'q', 'B', 'H', 'I', 'L', 'Q', 'f', 'd']

        # ------ python data types ------

        # map python types to internal reduction C code identifier
        if os.name == 'nt':
            self.python_type_map = {int: C_LONG_LONG, float: C_DOUBLE, bool: C_CHAR}
        else:
            self.python_type_map = {int: C_LONG, float: C_DOUBLE, bool: C_CHAR}

    # return Charm internal reducer type code and data ready to be sent to Charm
    def prepare(self, data, reducer, contributor):

        # nop reduction short circuit
        if (reducer is None) or (reducer == self.reducers.nop):
            return (self.reducers.nop, None, None)

        pyReducer = None
        if type(reducer) == tuple:
            op, py_red_func = reducer
            dt = type(data)
            if isinstance(data, np.ndarray) or isinstance(data, np.number):
                if not data.dtype.hasobject:
                    c_type = self.numpy_type_map[data.dtype.name]
                    charm_reducer_type = self.red_table[op][c_type]
                else:
                    pyReducer = py_red_func
            elif dt == array.array:
                c_type = self.array_type_map[data.typecode]
                charm_reducer_type = self.red_table[op][c_type]
            else:
                check_elems = data
                if not hasattr(data, '__len__'):
                    check_elems = [data]

                PYTHON_BASIC_TYPES = self.python_type_map.keys()
                t0 = type(check_elems[0])
                for elem in check_elems:
                    t = type(elem)
                    if (t != t0) or (t not in PYTHON_BASIC_TYPES):
                        pyReducer = py_red_func
                        break
                if pyReducer is None:
                    c_type = self.python_type_map[t0]
                    charm_reducer_type = self.red_table[op][c_type]
                    if not hasattr(data, '__len__'):
                        data = [data]
        else:
            pyReducer = reducer

        if pyReducer is None:
            return (charm_reducer_type, data, c_type)
        else:
            if not hasattr(pyReducer, 'hasPreprocess'):
                from .charm import Charm4PyError
                raise Charm4PyError("Please register reducer '" + reducer.__name__ + "' with addReducer")
            if pyReducer.hasPreprocess:
                data = pyReducer.preprocess(data, contributor)
            rednMsg = ({b"custom_reducer": pyReducer.__name__}, [data])
            # data for custom reducers is a custom reduction msg
            data = cPickle.dumps(rednMsg, self.charm.opts.PICKLE_PROTOCOL)
            return (self.charm.ReducerType.external_py, data, C_CHAR)

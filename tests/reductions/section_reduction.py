from charm4py import charm, Chare, Array, Reducer, Future, coro
import numpy as np

class TestVec(Chare):
    def __init__(self, op, myData):
        self.op = op
        self.data = myData

    def do_test( self, section, future ):
        if section:
            self.reduce( future, self.data, self.op, section )
        else:
            self.reduce( future, self.data, self.op )

class Test(Chare):
    def __init__(self, op, myData, section, doneFut):
        pass

def get_op_name( reducer_op ):
    for name, reducer_func in Reducer.__dict__.items():
        if reducer_func == reducer_op:
            return name

@coro
def test_op( done, op, vector_size, use_numpy = False ):
    if use_numpy:
        data = np.random.rand(vector_size)
    else:
        data = list(range(0,vector_size))

    finished_future = Future( 2 )
    chares = Array( TestVec, vector_size, args = [ op, data ] )
    chares.do_test( None, finished_future, awaitable = True ).get()
    section = chares[ 0 : vector_size ]
    section.do_test( section, finished_future )
    val1, val2 = finished_future.get()

    try:
        if use_numpy:
            assert np.isclose(val1, val2, atol = 1e-5).all()
        else:
            assert list(val1) == list(val2)
        print( f'[Main] Reduction with Reducer.{get_op_name(op)} passes.' )
        done(True)
    except AssertionError as e:
        print( f'[Main] Reduction with Reducer.{get_op_name(op)} is not correct.' )
        done( False )

@coro
def test_op_logical( done, op, vector_size, use_numpy = False ):
    if use_numpy:
        data = np.random.rand(vector_size)
        p = 0.1
        np.random.choice(a=[False, True], size=(1, vector_size), p=[p, 1-p] )
    else:
        data = list(range(0,vector_size))

    finished_future = Future( 2 )
    chares = Array( TestVec, vector_size, args = [ op, list(map(bool,range(0,vector_size))) ] )
    chares.do_test( None, finished_future )
    section = chares[ 0 : vector_size ]
    section.do_test( section, finished_future )
    val1, val2 = finished_future.get()
    try:
        assert list(val1) == list(val2)
        print( f'[Main] Reduction with Reducer.{get_op_name(op)} passes.' )
        done(True)
    except AssertionError as e:
        print( f'[Main] Reduction with Reducer.{get_op_name(op)} is not correct.' )
        done( False )

def main(args):

    num_tests = 14
    test_futures = [Future() for _ in range( num_tests )]

    # tests that when all chares participate in a section
    # that the answer is the same as when the entire array reduces.
    test_op( test_futures[0], Reducer.sum, 500 )
    test_op( test_futures[1], Reducer.product, 10 )
    test_op( test_futures[2], Reducer.min, 100 )
    test_op( test_futures[3], Reducer.max, 100 )
    test_op_logical( test_futures[4], Reducer.logical_and, 100)
    test_op_logical( test_futures[5], Reducer.logical_or, 100)
    test_op_logical( test_futures[6], Reducer.logical_xor, 100)

    # tests that when all chares participate in a section
    # that the answer is the same as when the entire array reduces,
    # the values contributed are numpy arrays.
    test_op( test_futures[7], Reducer.sum, 500, True )
    test_op( test_futures[8], Reducer.product, 10, True )
    test_op( test_futures[9], Reducer.min, 100, True )
    test_op( test_futures[10], Reducer.max, 100, True )
    test_op_logical( test_futures[11], Reducer.logical_and, 100, True)
    test_op_logical( test_futures[12], Reducer.logical_or, 100, True)
    test_op_logical( test_futures[13], Reducer.logical_xor, 100, True)


    passes = sum(map(lambda x: x.get(), test_futures))

    if passes == num_tests:
        print( 'All tests passed!' )
        exit()
    else:
        print( 'ERROR: Not all tests passed.' )
        exit(1)

charm.start(main)

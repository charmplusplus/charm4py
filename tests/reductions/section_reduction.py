from charm4py import charm, Chare, Array, Reducer, Future, coro

def pairwiseOp(op, data):
    pass
def elementwiseOp(op, data):
    pass

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
def test_op( done, op, vector_size ):
    finished_future = Future( 2 )
    chares = Array( TestVec, vector_size, args = [ op, list(range(0,vector_size)) ] )
    chares.do_test( None, finished_future )
    sum_section = chares[ 0 : vector_size ]
    sum_section.do_test( sum_section, finished_future )
    sum1, sum2 = finished_future.get()
    try:
        assert list(sum1) == list(sum2)
        done(True)
    except AssertionError as e:
        print( f'[Main] Reduction with Reducer.{get_op_name(op)} is not correct.' )
        done( False )

@coro
def test_op_logical( done, op, vector_size ):
    finished_future = Future( 2 )
    chares = Array( TestVec, vector_size, args = [ op, list(map(bool,range(0,vector_size))) ] )
    chares.do_test( None, finished_future )
    sum_section = chares[ 0 : vector_size ]
    sum_section.do_test( sum_section, finished_future )
    sum1, sum2 = finished_future.get()
    try:
        assert list(sum1) == list(sum2)
        done(True)
    except AssertionError as e:
        print( f'[Main] Reduction with Reducer.{get_op_name(op)} is not correct.' )
        done( False )




def main(args):

    num_tests = 7
    test_futures = [Future() for _ in range( num_tests )]

    test_op( test_futures[0], Reducer.sum, 500 )
    test_op( test_futures[1], Reducer.product, 10 )
    test_op( test_futures[2], Reducer.min, 100 )
    test_op( test_futures[3], Reducer.max, 100 )
    test_op_logical( test_futures[4], Reducer.logical_and, 100)
    test_op_logical( test_futures[5], Reducer.logical_or, 100)
    test_op_logical( test_futures[6], Reducer.logical_xor, 100)

    passes = sum(map(lambda x: x.get(), test_futures))

    if passes == num_tests:
        print( 'All tests passed!' )
        exit()
    else:
        print( 'ERROR: Not all tests passed.' )
        exit(1)

charm.start(main)

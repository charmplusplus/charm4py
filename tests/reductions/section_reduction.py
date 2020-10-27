from charm4py import charm, Chare, Array, Reducer, Future, coro
import numpy as np
import random


class TestVec(Chare):
    def __init__(self, op, myData):
        self.op = op
        self.data = myData

    def do_test(self, section, future):
        if section:
            self.reduce(future, self.data, self.op, section)
        else:
            self.reduce(future, self.data, self.op)


class Test(Chare):
    def __init__(self, op, myData, section, doneFut):
        pass


def get_op_name(reducer_op):
    for name, reducer_func in Reducer.__dict__.items():
        if reducer_func == reducer_op:
            return name


@coro
def test_op(done, op, vector_size, use_numpy=False):
    if vector_size > 1:
        if use_numpy:
            data = np.random.rand(vector_size)
        else:
            data = list(range(0, vector_size))
    else:
        data = random.uniform(0, 5)

    finished_future = Future(2)
    chares = Array(TestVec, vector_size, args=[op, data])
    chares.do_test(None, finished_future, awaitable=True).get()
    section = chares[0:vector_size]
    section.do_test(section, finished_future)
    val1, val2 = finished_future.get()

    try:
        if vector_size > 1:
            if use_numpy:
                assert np.isclose(val1, val2, atol=1e-5).all()
            else:
                assert list(val1) == list(val2)
        else:
            assert val1 == val2
        print('[Main] Reduction with Reducer.%s passes.' % get_op_name(op))
        done(True)
    except AssertionError:
        print('[Main] Reduction with Reducer.%s is not correct.' % get_op_name(op))
        done(False)


@coro
def test_op_logical(done, op, vector_size, use_numpy=False):
    if vector_size > 1:
        if use_numpy:
            data = np.random.rand(vector_size)
            p = 0.1
            data = np.random.choice(a=[False, True], size=(vector_size), p=[p, 1-p])
        else:
            data = list(map(bool, range(0, vector_size)))
    else:
        data = bool(random.randint(0, 1))

    finished_future = Future(2)
    chares = Array(TestVec, vector_size, args=[op, data])
    chares.do_test(None, finished_future)
    section = chares[0:vector_size]
    section.do_test(section, finished_future)
    val1, val2 = finished_future.get()

    try:
        if vector_size > 1:
            assert list(val1) == list(val2)
        else:
            assert val1 == val2
        print('[Main] Reduction with Reducer.%s passes.' % get_op_name(op))
        done(True)
    except AssertionError:
        print('[Main] Reduction with Reducer.%s is not correct.' % get_op_name(op))
        done(False)


def main(args):

    num_tests = 28
    test_futures = [Future() for _ in range(num_tests)]
    fut = iter(test_futures)

    # tests that when all chares participate in a section
    # that the answer is the same as when the entire array reduces.
    test_op(next(fut), Reducer.sum, 500)
    test_op(next(fut), Reducer.product, 10)
    test_op(next(fut), Reducer.min, 100)
    test_op(next(fut), Reducer.max, 100)
    test_op_logical(next(fut), Reducer.logical_and, 10)
    test_op_logical(next(fut), Reducer.logical_or, 10)
    test_op_logical(next(fut), Reducer.logical_xor, 10)

    # tests that when all chares participate in a section
    # that the answer is the same as when the entire array reduces,
    # the values contributed are numpy arrays.
    test_op(next(fut), Reducer.sum, 500, True)
    test_op(next(fut), Reducer.product, 10, True)
    test_op(next(fut), Reducer.min, 100, True)
    test_op(next(fut), Reducer.max, 100, True)
    test_op_logical(next(fut), Reducer.logical_and, 100, True)
    test_op_logical(next(fut), Reducer.logical_or, 100, True)
    test_op_logical(next(fut), Reducer.logical_xor, 100, True)

    # test that single-value reductions still work
    test_op(next(fut), Reducer.sum, 1, False)
    test_op(next(fut), Reducer.product, 1, False)
    test_op(next(fut), Reducer.min, 1, False)
    test_op(next(fut), Reducer.max, 1, False)
    test_op_logical(next(fut), Reducer.logical_and, 1, False)
    test_op_logical(next(fut), Reducer.logical_or, 1, False)
    test_op_logical(next(fut), Reducer.logical_xor, 1, False)

    # tests that when all chares participate in a section
    # that the answer is the same as when the entire array reduces,
    # # the values contributed are numpy arrays.
    test_op(next(fut), Reducer.sum, 1, True)
    test_op(next(fut), Reducer.product, 1, True)
    test_op(next(fut), Reducer.min, 1, True)
    test_op(next(fut), Reducer.max, 1, True)
    test_op_logical(next(fut), Reducer.logical_and, 1, True)
    test_op_logical(next(fut), Reducer.logical_or, 1, True)
    test_op_logical(next(fut), Reducer.logical_xor, 1, True)

    passes = sum(map(lambda x: x.get(), test_futures))

    if passes == num_tests:
        print('All tests passed!')
        exit()
    else:
        print('ERROR: Not all tests passed.')
        exit(1)


charm.start(main)

from charm4py import charm, Chare, Array, Reducer
import time
import numpy

charm.options.profiling = True

CHARES_PER_PE = 8
NUM_ITER = 5000
DATA_LEN = 20000  # number of values
VAL_CHECK = float(sum(range(DATA_LEN)))
mainProxy = None
NUM_CHARES = 0


def assert_almost_equal(actual, desired, tol):
    assert abs(actual - desired) <= tol


class Main(Chare):

    def __init__(self, args):
        charm.thisProxy.updateGlobals({'mainProxy': self.thisProxy,
                                       'NUM_CHARES': charm.numPes() * CHARES_PER_PE}, '__main__', awaitable=True).get()
        self.arrayProxy = Array(Test, NUM_CHARES)
        self.arrayProxy.run()
        self.startTime = time.time()

    def collectSum(self, result):
        assert_almost_equal(result.sum(), NUM_CHARES * VAL_CHECK, 0.05)
        self.arrayProxy.run()

    def done(self):
        print('Program done in', time.time() - self.startTime)
        charm.printStats()
        exit()


class Test(Chare):

    def __init__(self):
        self.data = numpy.arange(DATA_LEN, dtype='float64')
        self.reductions = 0

    def run(self):
        if self.reductions == NUM_ITER:
            self.contribute(None, None, mainProxy.done)
        else:
            self.contribute(self.data, Reducer.sum, mainProxy.collectSum)
            self.reductions += 1


charm.start(Main)

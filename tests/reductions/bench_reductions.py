import charm4py
from charm4py import charm, Chare, Array, Reducer
from charm4py import readonlies as ro
import time
import numpy

charm4py.Options.PROFILING = True

CHARES_PER_PE = 8
NUM_ITER = 5000
DATA_LEN = 20000  # number of values
VAL_CHECK = float(sum(range(DATA_LEN)))

def assert_almost_equal(actual, desired, tol):
  assert abs(actual - desired) <= tol

class Main(Chare):
  def __init__(self, args):

    ro.mainProxy = self.thisProxy
    ro.NUM_CHARES = charm.numPes() * CHARES_PER_PE
    ro.arrayProxy = Array(Test, ro.NUM_CHARES)
    ro.arrayProxy.run()
    self.startTime = time.time()

  def collectSum(self, result):
    assert_almost_equal(result.sum(), ro.NUM_CHARES * VAL_CHECK, 0.05)
    ro.arrayProxy.run()

  def done(self):
    print("Program done in", time.time() - self.startTime)
    charm.printStats()
    exit()

class Test(Chare):
  def __init__(self):

    self.data = numpy.arange(DATA_LEN, dtype='float64')
    self.reductions = 0

  def run(self):
    if self.reductions == NUM_ITER:
      self.contribute(None, None, ro.mainProxy.done)
    else:
      self.contribute(self.data, Reducer.sum, ro.mainProxy.collectSum)
      self.reductions += 1


charm.start(Main)

import charmpy
from charmpy import charm, Mainchare, Array, CkExit, CkNumPes, Reducer
from charmpy import readonlies as ro
import time
import numpy

charmpy.Options.PROFILING = True

CHARES_PER_PE = 8
NUM_ITER = 5000
DATA_LEN = 20000  # number of values
VAL_CHECK = float(sum(range(DATA_LEN)))

def assert_almost_equal(actual, desired, tol):
  assert abs(actual - desired) <= tol

class Main(Mainchare):
  def __init__(self, args):
    super(Main,self).__init__()

    ro.mainProxy = self.thisProxy
    ro.NUM_CHARES = CkNumPes() * CHARES_PER_PE
    ro.arrayProxy = charm.TestProxy.ckNew(ro.NUM_CHARES)
    ro.arrayProxy.run()
    self.startTime = time.time()

  def collectSum(self, result):
    assert_almost_equal(result.sum(), ro.NUM_CHARES * VAL_CHECK, 0.05)
    ro.arrayProxy.run()

  def done(self):
    print("Program done in", time.time() - self.startTime)
    charm.printStats()
    CkExit()

class Test(Array):
  def __init__(self):
    super(Test,self).__init__()

    self.data = numpy.arange(DATA_LEN, dtype='float64')
    self.reductions = 0

  def run(self):
    if self.reductions == NUM_ITER:
      self.contribute(None, None, ro.mainProxy.done)
    else:
      self.contribute(self.data, Reducer.sum, ro.mainProxy.collectSum)
      self.reductions += 1

charm.start()

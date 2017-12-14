from charmpy import charm, Mainchare, Array, Group, CkMyPe, CkNumPes, CkExit, ReadOnlies, CkAbort
from charmpy import Reducer

# utility methods for assertions
def assert_allclose(actual, desired, tol):
  assert len(actual) == len(desired)
  assert sum([(abs(actual[i] - v) <= tol) for i,v in enumerate(desired)]) == len(actual)

def assert_almost_equal(actual, desired, tol):
  assert abs(actual -desired) <= tol

ro = ReadOnlies()

class Main(Mainchare):
  def __init__(self, args):
    super(Main,self).__init__()

    self.expectedReductions = 7
    self.recvdReductions = 0

    ro.nDims = 1
    ro.ARRAY_SIZE = [10] * ro.nDims # 1-D array with 10 elements
    ro.firstIdx = [0] * ro.nDims
    ro.lastIdx = tuple([x-1 for x in ro.ARRAY_SIZE])

    nElements = 1
    for x in ro.ARRAY_SIZE: nElements *= x
    print("Running reduction example on " + str(CkNumPes()) + " processors for " + str(nElements) + " elements, array dims=" + str(ro.ARRAY_SIZE))
    ro.mainProxy = self.thisProxy
    ro.arrProxy = charm.TestProxy.ckNew(ro.ARRAY_SIZE)
    ro.groupProxy = charm.TestGroupProxy.ckNew()
    ro.arrProxy.doReduction()

  def done_int(self, reduction_result):
    assert reduction_result == 420, "Array-to-singleton sum_int reduction failed"
    print("[Main] All sum_int contributions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
        CkExit()

  def done_nop(self):
    print("[Main] All nop contributions received. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
        CkExit()

  def done_float(self, reduction_result):
    assert_allclose(reduction_result, [101.0, 134.0, 45.0], 1e-03)
    print("[Main] All sum_float contributions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
        CkExit()

  def done_array_to_array(self):
    print("[Main] All array-to-array contributions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      CkExit()

  def done_array_to_array_bcast(self):
    print("[Main] All array-to-array bcast contributions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      CkExit()

  def done_array_to_group(self):
    print("[Main] All array-to-group contributions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      CkExit()

  def done_array_to_group_bcast(self):
    print("[Main] All array-to-group bcast contributions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      CkExit()

class Test(Array):
  def __init__(self):
    super(Test,self).__init__()
    print("Test " + str(self.thisIndex) + " created on PE " + str(CkMyPe()))

  def doReduction(self):
    print("Test element " + str(self.thisIndex) + " on PE " + str(CkMyPe()) + " is starting its contributions.")
    # test contributing single int back to Main
    self.contribute(42, Reducer.sum, ro.mainProxy.done_int)
    # test contributing list of floats back to main
    num = [10.1, 13.4]
    self.contribute(num+[float(self.thisIndex[0])], Reducer.sum, ro.mainProxy.done_float)
    # test nop reduction to main
    self.contribute(None, Reducer.nop, ro.mainProxy.done_nop)
    # test contributing to Test[0]
    self.contribute(4.2, Reducer.sum, self.thisProxy[0].reductionTarget)
    # test contributing to Test (broadcast)
    self.contribute([4.2, 8.4], Reducer.sum, self.thisProxy.reductionTargetBcast)
    # test contributing to TestGroup[0]
    self.contribute(4, Reducer.sum, ro.groupProxy[0].reduceFromArray)
    # test contributing to TestGroup (broadcast)
    self.contribute([0, 8, 3], Reducer.sum, ro.groupProxy.reduceFromArrayBcast)

  def reductionTarget(self, reduction_result):
    assert self.thisIndex[0] == 0
    assert_almost_equal(reduction_result, 42.0, 1e-03)
    ro.mainProxy.done_array_to_array()

  def reductionTargetBcast(self, reduction_result):
    assert_allclose(reduction_result, [42.0, 84.0], 1e-03)
    self.contribute(None, None, ro.mainProxy.done_array_to_array_bcast)

class TestGroup(Group):
  def __init__(self):
    super(TestGroup,self).__init__()
    print("TestGroup " + str(self.thisIndex) + " created on PE " + str(CkMyPe()))

  def reduceFromArray(self, reduction_result):
    assert self.thisIndex == 0
    assert reduction_result == 40, "Array-to-group sum_int reduction failed."
    ro.mainProxy.done_array_to_group()

  def reduceFromArrayBcast(self, reduction_result):
    assert reduction_result == [0, 80, 30], "Array-to-group bcast sum_int reduction failed."
    self.contribute(None, None, ro.mainProxy.done_array_to_group_bcast)

# ---- start charm ----
charm.start([Main,Test,TestGroup])

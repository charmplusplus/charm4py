from charmpy import charm, Mainchare, Array, Group, CkMyPe, CkNumPes, CkExit, ReadOnlies, CkAbort

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
    print("Running reduction example on " + str(CkNumPes()) + " processors")
    ro.mainProxy = self.thisProxy
    ro.groupProxy = charm.TestGroupProxy.ckNew()
    # create an array to test group-to-array reductions
    ro.arrayProxy = charm.TestArrayProxy.ckNew(ro.ARRAY_SIZE)
    ro.groupProxy.doReduction()

  def done_int(self, reduction_result):
    assert reduction_result == 42*CkNumPes(), "Group-to-singleton sum_int reduction failed"
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
    expected_result = [x*CkNumPes() for x in [10.1, 13.4]]
    indices_sum = (CkNumPes() * (CkNumPes() - 1))/2
    expected_result += [float(indices_sum)]
    assert_allclose(reduction_result, expected_result, 1e-03)
    print("[Main] All sum_float contributions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
        CkExit()

  def done_group_to_array(self):
    print("[Main] All group-to-array contributions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      CkExit()

  def done_group_to_array_bcast(self):
    print("[Main] All group-to-array bcast contributions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      CkExit()

  def done_group_to_group(self):
    print("[Main] All group-to-group contributions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      CkExit()

  def done_group_to_group_bcast(self):
    print("[Main] All group-to-group bcast contributions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      CkExit()

class TestGroup(Group):
  def __init__(self):
    super(TestGroup,self).__init__()
    print("TestGroup " + str(self.thisIndex) + " created on PE " + str(CkMyPe()))

  def doReduction(self):
    print("TestGroup element on PE " + str(CkMyPe()) + " is starting its contributions.")
    # test contributing single int back to Main
    self.contribute(42, charm.ReducerType.sum_int, ro.mainProxy.done_int)
    # test contributing list of floats back to Main
    num = [10.1, 13.4]
    self.contribute(num+[float(self.thisIndex)], charm.ReducerType.sum_float, ro.mainProxy.done_float)
    # test nop reduction to main
    self.contribute(None, charm.ReducerType.nop, ro.mainProxy.done_nop)
    # test contributing to TestArray[0]
    self.contribute([4.2, 13.1], charm.ReducerType.sum_float, ro.arrayProxy[0].reduceGroupToArray)
    # test contributing to TestArray (broadcast)
    self.contribute(-4, charm.ReducerType.sum_int, ro.arrayProxy.reduceGroupToArrayBcast)
    # test contributing to TestGroup[0]
    self.contribute([5, 7, -3, 0], charm.ReducerType.sum_int, self.thisProxy[0].reduceGroupToGroup)
    # test contributing to TestGroup (broadcast)
    self.contribute(-4.2, charm.ReducerType.sum_double, self.thisProxy.reduceGroupToGroupBcast)

  def reduceGroupToGroup(self, reduction_result):
    assert self.thisIndex == 0
    assert reduction_result == [CkNumPes()*x for x in [5, 7, -3, 0]], "Group-to-group reduction failed."
    ro.mainProxy.done_group_to_group()

  def reduceGroupToGroupBcast(self, reduction_result):
    assert_almost_equal(reduction_result, -4.2*CkNumPes(), 1e-03)
    self.contribute(None, charm.ReducerType.nop, ro.mainProxy.done_group_to_group_bcast)


class TestArray(Array):
  def __init__(self):
    super(TestArray,self).__init__()
    print("TestArray " + str(self.thisIndex) + " created on PE " + str(CkMyPe()))

  def reduceGroupToArray(self, reduction_result):
    assert self.thisIndex[0] == 0
    assert_allclose(reduction_result, [CkNumPes()*x for x in [4.2, 13.1]], 1e-03)
    ro.mainProxy.done_group_to_array()

  def reduceGroupToArrayBcast(self, reduction_result):
    assert reduction_result == -4*CkNumPes(), "Group-to-array bcast reduction failed."
    self.contribute(None, charm.ReducerType.nop, ro.mainProxy.done_group_to_array_bcast)

# ---- start charm ----
charm.start([Main,TestArray,TestGroup])

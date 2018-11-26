from charm4py import charm, Chare, Array, Group, Reducer
from charm4py import readonlies as ro

# utility methods for assertions
def assert_allclose(actual, desired, tol):
  assert len(actual) == len(desired)
  assert sum([(abs(actual[i] - v) <= tol) for i,v in enumerate(desired)]) == len(actual)

def assert_almost_equal(actual, desired, tol):
  assert abs(actual -desired) <= tol

class Main(Chare):
  def __init__(self, args):

    self.expectedReductions = 7
    self.recvdReductions = 0

    ro.nDims = 1
    ro.ARRAY_SIZE = [10] * ro.nDims # 1-D array with 10 elements
    ro.firstIdx = [0] * ro.nDims
    ro.lastIdx = tuple([x-1 for x in ro.ARRAY_SIZE])

    nElements = 1
    for x in ro.ARRAY_SIZE: nElements *= x
    print("Running reduction example on " + str(charm.numPes()) + " processors")
    ro.mainProxy = self.thisProxy
    ro.groupProxy = Group(TestGroup)
    # create an array to test group-to-array reductions
    ro.arrayProxy = Array(TestArray, ro.ARRAY_SIZE)
    ro.groupProxy.doReduction()

  def done_int(self, reduction_result):
    assert reduction_result == 42*charm.numPes(), "Group-to-singleton sum_int reduction failed"
    print("[Main] All sum_int contributions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
        exit()

  def done_nop(self):
    print("[Main] All nop contributions received. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
        exit()

  def done_float(self, reduction_result):
    expected_result = [x*charm.numPes() for x in [10.1, 13.4]]
    indices_sum = (charm.numPes() * (charm.numPes() - 1))/2
    expected_result += [float(indices_sum)]
    assert_allclose(reduction_result, expected_result, 1e-03)
    print("[Main] All sum_float contributions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
        exit()

  def done_group_to_array(self):
    print("[Main] All group-to-array contributions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      exit()

  def done_group_to_array_bcast(self):
    print("[Main] All group-to-array bcast contributions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      exit()

  def done_group_to_group(self):
    print("[Main] All group-to-group contributions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      exit()

  def done_group_to_group_bcast(self):
    print("[Main] All group-to-group bcast contributions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      exit()

class TestGroup(Chare):
  def __init__(self):
    print("TestGroup " + str(self.thisIndex) + " created on PE " + str(charm.myPe()))

  def doReduction(self):
    print("TestGroup element on PE " + str(charm.myPe()) + " is starting its contributions.")
    # test contributing single int back to Main
    self.contribute(42, Reducer.sum, ro.mainProxy.done_int)
    # test contributing list of floats back to Main
    num = [10.1, 13.4]
    self.contribute(num+[float(self.thisIndex)], Reducer.sum, ro.mainProxy.done_float)
    # test nop reduction to main
    self.contribute(None, Reducer.nop, ro.mainProxy.done_nop)
    # test contributing to TestArray[0]
    self.contribute([4.2, 13.1], Reducer.sum, ro.arrayProxy[0].reduceGroupToArray)
    # test contributing to TestArray (broadcast)
    self.contribute(-4, Reducer.sum, ro.arrayProxy.reduceGroupToArrayBcast)
    # test contributing to TestGroup[0]
    self.contribute([5, 7, -3, 0], Reducer.sum, self.thisProxy[0].reduceGroupToGroup)
    # test contributing to TestGroup (broadcast)
    self.contribute(-4.2, Reducer.sum, self.thisProxy.reduceGroupToGroupBcast)

  def reduceGroupToGroup(self, reduction_result):
    assert self.thisIndex == 0
    assert list(reduction_result) == [charm.numPes()*x for x in [5, 7, -3, 0]], "Group-to-group reduction failed."
    ro.mainProxy.done_group_to_group()

  def reduceGroupToGroupBcast(self, reduction_result):
    assert_almost_equal(reduction_result, -4.2*charm.numPes(), 1e-03)
    self.contribute(None, None, ro.mainProxy.done_group_to_group_bcast)


class TestArray(Chare):
  def __init__(self):
    print("TestArray " + str(self.thisIndex) + " created on PE " + str(charm.myPe()))

  def reduceGroupToArray(self, reduction_result):
    assert self.thisIndex[0] == 0
    assert_allclose(reduction_result, [charm.numPes()*x for x in [4.2, 13.1]], 1e-03)
    ro.mainProxy.done_group_to_array()

  def reduceGroupToArrayBcast(self, reduction_result):
    assert reduction_result == -4*charm.numPes(), "Group-to-array bcast reduction failed."
    self.contribute(None, None, ro.mainProxy.done_group_to_array_bcast)


charm.start(Main)

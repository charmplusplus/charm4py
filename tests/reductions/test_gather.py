from charm4py import charm, Chare, Array, Group, Reducer
from charm4py import readonlies as ro


class Main(Chare):
  def __init__(self, args):

    self.recvdReductions = 0
    self.expectedReductions = 5

    ro.nDims = 1
    ro.ARRAY_SIZE = [10] * ro.nDims
    ro.firstIdx = [0] * ro.nDims
    ro.lastIdx = tuple([x-1 for x in ro.ARRAY_SIZE])

    self.nElements = 1
    for x in ro.ARRAY_SIZE: self.nElements *= x
    print("Running gather example on " + str(charm.numPes()) + " processors for " + str(self.nElements) + " elements, array dims=" + str(ro.ARRAY_SIZE))
    ro.mainProxy = self.thisProxy
    ro.arrProxy = Array(Test, ro.ARRAY_SIZE)
    ro.grpProxy = Group(TestGroup)
    ro.arrProxy.doGather()
    ro.grpProxy.doGather()
    red_future = charm.createFuture()
    ro.arrProxy.doGather(red_future)
    self.done_gather_single(red_future.get())

  def done_gather_single(self, result):
    gather_arr_indices = list(range(self.nElements))
    gather_grp_indices = list(range(charm.numPes()))
    assert result == gather_arr_indices or result == gather_grp_indices, "Gather single elements failed."
    print("[Main] Gather collective for single elements done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      exit()

  def done_gather_array(self, result):
    gather_arr_indices = [tuple([i]) for i in range(self.nElements)]
    gather_grp_indices = [[i, 42] for i in range(charm.numPes())]
    assert result == gather_arr_indices or result == gather_grp_indices, "Gather arrays failed."
    print("[Main] Gather collective for arrays done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      exit()

class Test(Chare):
  def __init__(self):
    print("Test " + str(self.thisIndex) + " created on PE " + str(charm.myPe()))

  def doGather(self, red_future=None):
    if red_future is None:
      # gather single elements
      self.contribute(self.thisIndex[0], Reducer.gather, ro.mainProxy.done_gather_single)
      # gather arrays
      self.contribute(self.thisIndex, Reducer.gather, ro.mainProxy.done_gather_array)
    else:
      self.contribute(self.thisIndex[0], Reducer.gather, red_future)

class TestGroup(Chare):
  def __init__(self):
    print("TestGroup " + str(self.thisIndex) + " created on PE " + str(charm.myPe()))

  def doGather(self):
    # gather single elements
    self.contribute(self.thisIndex, Reducer.gather, ro.mainProxy.done_gather_single)
    # gather arrays
    self.contribute([self.thisIndex, 42], Reducer.gather, ro.mainProxy.done_gather_array)


charm.start(Main)

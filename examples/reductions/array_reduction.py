from charmpy import charm, Mainchare, Array, Group, CkMyPe, CkNumPes, CkExit, ReadOnlies, CkAbort

ro = ReadOnlies()

class Main(Mainchare):
  def __init__(self, args):
    super(Main,self).__init__()

    self.expectedReductions = 7
    self.recvdReductions = 0
    self.groupBcast = 0 #TODO: remove after adding Group contribute support

    ro.nDims = 1
    ro.ARRAY_SIZE = [10] * ro.nDims # 1-D array with 10 elements
    ro.firstIdx = [0] * ro.nDims
    ro.lastIdx = tuple([x-1 for x in ro.ARRAY_SIZE])

    nElements = 1
    for x in ro.ARRAY_SIZE: nElements *= x
    print "Running reduction example on", CkNumPes(), "processors for", nElements, "elements, array dims=", ro.ARRAY_SIZE
    ro.mainProxy = self.thisProxy
    arrProxy = charm.TestProxy.ckNew(ro.ARRAY_SIZE)
    ro.groupProxy = charm.TestGroupProxy.ckNew()
    arrProxy.doReduction()

  def done_int(self, reduction_result):
    print "[Main] All sum_int contributions done"
    print "[Main] Total sum: ", reduction_result
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
        CkExit()

  def done_nop(self):
    print "[Main] All nop contributions received"
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
        CkExit()

  def done_float(self, reduction_result):
    print "[Main] All sum_float contributions done"
    print "[Main] Total sum: ", reduction_result
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
        CkExit()

  def done_array_to_array(self):
    print "[Main] All array-to-array contributions done"
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      CkExit()

  def done_array_to_array_bcast(self):
    print "[Main] All array-to-array bcast contributions done"
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      CkExit()

  def done_array_to_group(self):
    print "[Main] All array-to-group contributions done"
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      CkExit()

  def done_array_to_group_bcast(self):
    self.groupBcast += 1
    if self.groupBcast == CkNumPes():
      print "[Main] All array-to-group bcast contributions done"
      self.recvdReductions += 1
      if (self.recvdReductions >= self.expectedReductions):
        CkExit()

class Test(Array):
  def __init__(self):
    super(Test,self).__init__()
    print "Test", self.thisIndex, "created on PE", CkMyPe()

  def doReduction(self):
    print "Test element", self.thisIndex, "on PE", CkMyPe(), " is starting its contributions."
    # test contributing single int back to Main
    self.contribute(42, charm.ReducerType.sum_int, Main.done_int, ro.mainProxy)
    # test contributing list of floats back to main
    num = [10.1, 13.4]
    self.contribute(num+[float(self.thisIndex[0])], charm.ReducerType.sum_float, Main.done_float, ro.mainProxy)
    # test nop reduction to main
    self.contribute(None, charm.ReducerType.nop, Main.done_nop, ro.mainProxy)
    # test contributing to Test[0]
    self.contribute(4.2, charm.ReducerType.sum_double, Test.reductionTarget, self.thisProxy[(0,)])
    # test contributing to Test (broadcast)
    self.contribute([4.2, 8.4], charm.ReducerType.sum_double, Test.reductionTargetBcast, self.thisProxy)
    # test contributing to TestGroup[0]
    self.contribute(4, charm.ReducerType.sum_int, TestGroup.reduceFromArray, ro.groupProxy[0])
    # test contributing to TestGroup (broadcast)
    self.contribute([0, 8, 3], charm.ReducerType.sum_int, TestGroup.reduceFromArrayBcast, ro.groupProxy)

  def reductionTarget(self, reduction_result):
    assert(self.thisIndex[0] == 0)
    print "[Test ", self.thisIndex, "] Total sum: ", reduction_result
    ro.mainProxy.done_array_to_array()

  def reductionTargetBcast(self, reduction_result):
    print "[Test ", self.thisIndex, "] Total sum: ", reduction_result
    self.contribute(None, charm.ReducerType.nop, Main.done_array_to_array_bcast, ro.mainProxy)

class TestGroup(Group):
  def __init__(self):
    super(TestGroup,self).__init__()
    print "TestGroup", self.thisIndex, "created on PE", CkMyPe()

  def reduceFromArray(self, reduction_result):
    print "[TestGroup ", self.thisIndex, "] Total sum: ", reduction_result
    ro.mainProxy.done_array_to_group()

  def reduceFromArrayBcast(self, reduction_result):
    print "[TestGroup ", self.thisIndex, "] Total sum: ", reduction_result
    #TODO: add contribute support for groups
    #self.contribute(None, charm.ReducerType.nop, Main.done_array_to_group_bcast, ro.mainProxy)
    ro.mainProxy.done_array_to_group_bcast()

# ---- start charm ----
charm.start([Main,Test,TestGroup])

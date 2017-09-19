from charmpy import charm, Mainchare, Array, CkMyPe, CkNumPes, CkExit, ReadOnlies, CkAbort

ro = ReadOnlies()

class Main(Mainchare):
  def __init__(self, args):
    super(Main,self).__init__()

    self.expectedReductions = 3
    self.recvdReductions = 0

    ro.nDims = 1
    ro.ARRAY_SIZE = [10] * ro.nDims # 1-D array with 10 elements
    ro.firstIdx = [0] * ro.nDims
    ro.lastIdx = tuple([x-1 for x in ro.ARRAY_SIZE])

    nElements = 1
    for x in ro.ARRAY_SIZE: nElements *= x
    print "Running reduction example on", CkNumPes(), "processors for", nElements, "elements, array dims=", ro.ARRAY_SIZE
    ro.mainProxy = self.thisProxy
    arrProxy = charm.TestProxy.ckNew(ro.ARRAY_SIZE)
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
    #self.contribute(42, charm.ReducerType.sum_int, Test.reductionTarget, self.thisProxy[(0,)])

  def reductionTarget(reduction_result):
    CkExit()

# ---- start charm ----
charm.start([Main,Test])

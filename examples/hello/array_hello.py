from charmpy import charm, Mainchare, Array, CkMyPe, CkNumPes, CkExit, ReadOnlies, CkAbort

ro = ReadOnlies()

class Main(Mainchare):
  def __init__(self, args):
    super(Main,self).__init__()

    if len(args) <= 1:
      args = [None,3,2]  # default: 3 dimensions of size 2 each
    elif len(args) != 3:
      CkAbort("Usage : python array_hello.py [<num_dimensions> <array_size>]")

    ro.nDims = int(args[1])
    ro.ARRAY_SIZE = [int(args[2])] * ro.nDims
    ro.firstIdx = [0] * ro.nDims
    ro.lastIdx = tuple([x-1 for x in ro.ARRAY_SIZE])

    nElements = 1
    for x in ro.ARRAY_SIZE: nElements *= x
    print "Running Hello on", CkNumPes(), "processors for", nElements, "elements, array dims=", ro.ARRAY_SIZE
    ro.mainProxy = self.thisProxy
    arrProxy = charm.HelloProxy.ckNew(ro.ARRAY_SIZE)
    arrProxy[ro.firstIdx].SayHi(17)

  def done(self):
    print "All done"
    CkExit()

class Hello(Array):
  def __init__(self):
    super(Hello,self).__init__()
    print "Hello", self.thisIndex, "created on PE", CkMyPe()

  def SayHi(self, hiNo):
    print "Hi[", hiNo, "] from element", self.thisIndex, "on PE", CkMyPe()
    if self.thisIndex == ro.lastIdx:
      ro.mainProxy.done()
    else:
      nextIndex = list(self.thisIndex)
      for i in range(ro.nDims-1,-1,-1):
        nextIndex[i] = (nextIndex[i] + 1) % ro.ARRAY_SIZE[i]
        if nextIndex[i] != 0: break
      return self.thisProxy[nextIndex].SayHi(hiNo+1)

# ---- start charm ----
charm.start([Main,Hello])

from charmpy import charm, Mainchare, Array, CkMyPe, CkNumPes, CkExit, CkAbort
from charmpy import readonlies as ro


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
    print("Running Hello on " + str(CkNumPes()) + " processors for " + str(nElements) + " elements, array dims=" + str(ro.ARRAY_SIZE))
    ro.mainProxy = self.thisProxy
    arrProxy = charm.HelloProxy.ckNew(ro.ARRAY_SIZE)
    arrProxy[ro.firstIdx].SayHi(17)

  def done(self):
    print("All done")
    CkExit()

class Hello(Array):
  def __init__(self):
    super(Hello,self).__init__()
    print("Hello " + str(self.thisIndex) + " created on PE " + str(CkMyPe()))

  def SayHi(self, hiNo):
    print("Hi[" + str(hiNo) + "] from element " + str(self.thisIndex) + " on PE " + str(CkMyPe()))
    if self.thisIndex == ro.lastIdx:
      ro.mainProxy.done()
    else:
      nextIndex = list(self.thisIndex)
      for i in range(ro.nDims-1,-1,-1):
        nextIndex[i] = (nextIndex[i] + 1) % ro.ARRAY_SIZE[i]
        if nextIndex[i] != 0: break
      return self.thisProxy[nextIndex].SayHi(hiNo+1)

# ---- start charm ----
charm.start()

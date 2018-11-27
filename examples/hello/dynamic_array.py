from charm4py import charm, Chare, Array, Reducer
from charm4py import readonlies as ro
import itertools


class Main(Chare):
  def __init__(self, args):

    if len(args) <= 1:
      args = [None,2,3]  # default: 2 dimensions of size 3
    elif len(args) != 3:
      charm.abort("Usage : python array_hello.py [<num_dimensions> <array_size>]")

    ro.nDims = int(args[1])
    ro.ARRAY_SIZE = [int(args[2])] * ro.nDims
    ro.firstIdx = [0] * ro.nDims
    ro.lastIdx = tuple([x-1 for x in ro.ARRAY_SIZE])

    self.nElements = 1
    for x in ro.ARRAY_SIZE: self.nElements *= x
    print("Running Hello on " + str(charm.numPes()) + " processors for " + str(self.nElements) + " elements")
    ro.mainProxy = self.thisProxy
    self.arrProxy = Array(Hello, ndims=ro.nDims)
    print("Created array proxy")
    indices = list(itertools.product(range(ro.ARRAY_SIZE[0]), repeat=ro.nDims))
    assert len(indices) == self.nElements
    for i in indices:
      self.arrProxy.ckInsert(i, [42, "testing"])

    self.arrProxy.ckDoneInserting()
    self.arrProxy[ro.firstIdx].SayHi(17)

  def done(self):
    print("Ring messaging done. Testing reduction.")
    self.arrProxy.TestReduction()

  def doneReduction(self, result):
    assert result == 1*self.nElements, "Reduction for dynamic array insertion failed."
    print("All done.")
    exit()

class Hello(Chare):
  def __init__(self, arg0, arg1):
    assert [arg0, arg1] == [42, "testing"], "Constructor args for dynamic array insertion failed."
    print("Hello " + str(self.thisIndex) + " created on PE " + str(charm.myPe()))

  def SayHi(self, hiNo):
    print("Hi[" + str(hiNo) + "] from element " + str(self.thisIndex) + " on PE " + str(charm.myPe()))
    if self.thisIndex == ro.lastIdx:
      ro.mainProxy.done()
    else:
      nextIndex = list(self.thisIndex)
      for i in range(ro.nDims-1,-1,-1):
        nextIndex[i] = (nextIndex[i] + 1) % ro.ARRAY_SIZE[i]
        if nextIndex[i] != 0: break
      return self.thisProxy[nextIndex].SayHi(hiNo+1)

  def TestReduction(self):
    self.contribute(1, Reducer.sum, ro.mainProxy.doneReduction)


charm.start(Main)

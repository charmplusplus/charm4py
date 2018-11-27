from charm4py import charm, Chare, Array
from charm4py import readonlies as ro


class Hello(Chare):
  def __init__(self):
    print("Hello " + str(self.thisIndex) + " created on PE " + str(charm.myPe()))

  def SayHi(self, hiNo):
    print("Hi[" + str(hiNo) + "] from element " + str(self.thisIndex) + " on PE " + str(charm.myPe()))
    if self.thisIndex == ro.lastIdx:
      print("All done")
      exit()
    else:
      nextIndex = list(self.thisIndex)
      for i in range(ro.nDims-1,-1,-1):
        nextIndex[i] = (nextIndex[i] + 1) % ro.ARRAY_SIZE[i]
        if nextIndex[i] != 0: break
      return self.thisProxy[nextIndex].SayHi(hiNo+1)


def main(args):

  if len(args) <= 1:
    args = [None,3,2]  # default: 3 dimensions of size 2 each
  elif len(args) != 3:
    charm.abort("Usage : python array_hello.py [<num_dimensions> <array_size>]")

  ro.nDims = int(args[1])
  ro.ARRAY_SIZE = [int(args[2])] * ro.nDims
  ro.firstIdx = [0] * ro.nDims
  ro.lastIdx = tuple([x-1 for x in ro.ARRAY_SIZE])

  nElements = 1
  for x in ro.ARRAY_SIZE: nElements *= x
  print("Running Hello on " + str(charm.numPes()) + " processors for " + str(nElements) + " elements, array dims=" + str(ro.ARRAY_SIZE))
  arrProxy = Array(Hello, ro.ARRAY_SIZE)
  arrProxy[ro.firstIdx].SayHi(17)


charm.start(main)

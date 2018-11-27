from charm4py import charm, Chare, Array, Group
from charm4py import readonlies as ro
import numpy


class Hello(Chare):
  def __init__(self, arg0, arg1, arg2):
    assert [arg0, arg1, arg2] == [1, "test", [4.2, -9]], "Construtor args for array failed."
    print("Hello " + str(self.thisIndex) + " created on PE " + str(charm.myPe()))

  def SayHi(self, hiNo):
    print("Hi[" + str(hiNo) + "] from element " + str(self.thisIndex) + " on PE " + str(charm.myPe()))
    if self.thisIndex == ro.lastIdx:
      print("Array done")
      ro.grpProxy[0].SayHi(17)
    else:
      nextIndex = list(self.thisIndex)
      for i in range(ro.nDims-1,-1,-1):
        nextIndex[i] = (nextIndex[i] + 1) % ro.ARRAY_SIZE[i]
        if nextIndex[i] != 0: break
      return self.thisProxy[nextIndex].SayHi(hiNo+1)


class HelloGroup(Chare):
  def __init__(self, arg0, arg1, arg2, arg3):
    assert [arg0, arg1, arg2] == [1, "test", [4.2, -9]], "Constructor args for groups failed."
    assert arg3.all() == numpy.full((3,5), 4.2).all(), "Numpy constructor arg for groups failed."
    print("HelloGroup " + str(self.thisIndex) + " created")

  def SayHi(self, hiNo):
    print("Hi[" + str(hiNo) + "] from element " + str(self.thisIndex))
    if self.thisIndex + 1 < charm.numPes():
      # Pass the hello on:
      self.thisProxy[self.thisIndex+1].SayHi(hiNo+1)
    else:
      print("All done")
      exit()


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
  arrProxy = Array(Hello, ro.ARRAY_SIZE, args=[1, "test", [4.2, -9]])
  ro.grpProxy = Group(HelloGroup, args=[1, "test", [4.2, -9], numpy.full((3,5), 4.2)])

  arrProxy[ro.firstIdx].SayHi(17)


charm.start(main)

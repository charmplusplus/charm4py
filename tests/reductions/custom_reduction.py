from charm4py import charm, Chare, Array, Group, Reducer
from charm4py import readonlies as ro

def myReducer(contribs):
  result = []
  result.append(sum([c[0] for c in contribs]))
  result.append(max([c[1] for c in contribs]))
  result.append(min([c[2] for c in contribs]))
  return result

Reducer.addReducer(myReducer)

class Main(Chare):
  def __init__(self, args):
    self.recvdReductions = 0
    self.expectedReductions = 4

    ro.nDims = 1
    ro.ARRAY_SIZE = [10] * ro.nDims
    ro.firstIdx = [0] * ro.nDims
    ro.lastIdx = tuple([x-1 for x in ro.ARRAY_SIZE])

    self.nElements = 1
    for x in ro.ARRAY_SIZE: self.nElements *= x
    print("Running reduction example on " + str(charm.numPes()) + " processors for " + str(self.nElements) + " elements, array dims=" + str(ro.ARRAY_SIZE))
    ro.mainProxy = self.thisProxy
    ro.arrProxy = Array(Test, ro.ARRAY_SIZE)
    ro.arrProxy.doReduction()

  def done_charm_builtin(self, result):
    sum_indices = (self.nElements*(self.nElements-1))/2
    assert list(result) == [10, sum_indices], "Built-in Charm sum_int reduction failed"
    print("[Main] All Charm builtin reductions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      exit()

  def done_python_builtin(self, result):
    sum_indices = (self.nElements*(self.nElements-1))/2
    assert type(result) == MyObject
    assert result.value == sum_indices or result.value == 0, "Built-in Python _sum or _product reduction failed"
    print("[Main] All Python builtin reductions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      exit()

  def done_python_custom(self, result):
    assert result == [10, ro.lastIdx[0], 0], "Custom Python myReduce failed"
    print("[Main] All Python custom reductions done. Test passed")
    self.recvdReductions += 1
    if (self.recvdReductions >= self.expectedReductions):
      exit()

class MyObject(object):
  def __init__(self, n):
    self.value = n

  def __add__(self, other):
    return MyObject(self.value+other.value)

  def __mul__(self, other):
    return MyObject(self.value*other.value)

  def __radd__(self, other):
    if other == 0:
      return self
    else:
      return self.__add__(other)

class Test(Chare):
  def __init__(self):
    print("Test " + str(self.thisIndex) + " created on PE " + str(charm.myPe()))

  def doReduction(self):
    # test contributing using built-in Charm reducer
    self.contribute([1,self.thisIndex[0]], Reducer.sum, ro.mainProxy.done_charm_builtin)
    a = MyObject(self.thisIndex[0])
    # test contributing using built-in Python reducer
    self.contribute(a, Reducer.sum, ro.mainProxy.done_python_builtin)
    # test product reducer
    self.contribute(a, Reducer.product, ro.mainProxy.done_python_builtin)
    # test contributing using custom Python reducer
    self.contribute([1,self.thisIndex[0],self.thisIndex[0]], Reducer.myReducer, ro.mainProxy.done_python_custom)


charm.start(Main)

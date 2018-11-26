from charm4py import charm, Chare, Group, Array, Reducer
from charm4py import readonlies as ro
import numpy
import math

MAX_ITER = 100
all_created = False
arrayElemHomeMap = {} # array elem index -> original pe

class Controller(Chare):

    def __init__(self):
        pass

    def arrayElemsCreated(self, home_pes):
        global all_created, arrayElemHomeMap
        all_created = True
        for i in range(len(home_pes)): arrayElemHomeMap[i] = home_pes[i]
        self.contribute(None, None, ro.array.start)


class Test(Chare):

    def __init__(self):
        assert(not all_created) # makes sure constructor is only called for creation, not migration
        self.iteration = 0
        self.originalPe = charm.myPe()
        self.data = numpy.arange(100, dtype='int64') * (self.originalPe + 1)
        # notify controllers that array elements are created and pass home PE of every element
        self.contribute(charm.myPe(), Reducer.gather, ro.controllers.arrayElemsCreated)

    def start(self):
        if self.thisIndex == (0,) and self.iteration % 20 == 0: print("Iteration " + str(self.iteration))
        self.check()
        A = numpy.arange(1000, dtype='float64')
        work = 1000 * int(round(math.log(charm.myPe() + 1) + 1))  # elements in higher PEs do more work
        for i in range(work): A += 1.33
        self.iteration += 1
        if self.iteration == MAX_ITER:
            self.contribute(None, None, charm.thisProxy[0].exit)
        elif self.iteration % 20 == 0:
            self.AtSync()
        else:
            self.thisProxy[self.thisIndex].start()

    def resumeFromSync(self):
        self.start()

    def check(self):  # check that my attributes haven't changed as a result of migrating
        assert(self.originalPe == arrayElemHomeMap[self.thisIndex[0]])
        v = numpy.arange(100, dtype='int64') * (self.originalPe + 1)
        numpy.testing.assert_allclose(self.data, v)


def main(args):
    ro.controllers = Group(Controller)
    ro.array       = Array(Test, charm.numPes() * 4)


charm.start(main)

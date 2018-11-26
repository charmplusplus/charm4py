import charm4py
from charm4py import charm, Chare, Array, Reducer
from charm4py import readonlies as ro
import time
import array
import numpy
from numpy.testing import assert_allclose
from collections import defaultdict

charm4py.Options.PROFILING = True
charm4py.Options.LOCAL_MSG_OPTIM = False

MAX_ITER = 50
DATA_LEN = 15000        # number of doubles
CHARES_PER_PE = 10

class Main(Chare):

    def __init__(self, args):
        ro.mainProxy = self.thisProxy
        ro.testProxy = Array(Test, charm.numPes() * CHARES_PER_PE)

    def start(self):
        self.iterations = 0
        self.startTime = time.time()
        ro.testProxy.doIteration()

    def iterationComplete(self):
        if self.iterations % 10 == 0: print("Iteration", self.iterations, "complete")
        self.iterations += 1
        if self.iterations == MAX_ITER:
            print("Program done. Total time =", time.time() - self.startTime)
            charm.printStats()
            exit()
        else:
            ro.testProxy.doIteration()


class Test(Chare):
    def __init__(self):

        self.x = numpy.arange(DATA_LEN, dtype='float64')
        y = self.x * (self.thisIndex[0] + 1)

        self.S1 = y.tobytes()
        self.S2 = array.array('d', y)
        self.S3 = y

        self.msgsRcvd = 0

        self.contribute(charm.myPe(), Reducer.gather, self.thisProxy.recvLocations)

    def recvLocations(self, locations):
        loc = defaultdict(list)
        for chare_idx, pe in enumerate(locations): loc[pe].append(chare_idx)
        myPe = charm.myPe()
        myPos = loc[myPe].index(self.thisIndex[0])
        # i-th chare in a PE sends to i-th chare in PE-1 and PE+1 and to itself
        nb1 = self.thisProxy[loc[(myPe - 1) % charm.numPes()][myPos]]
        nb2 = self.thisProxy[loc[(myPe + 1) % charm.numPes()][myPos]]
        self.nbs = [nb1, nb2, self.thisProxy[self.thisIndex]]
        self.contribute(None, None, ro.mainProxy.start)

    def doIteration(self):
        for nb in self.nbs:
            nb.recvData(self.thisIndex, self.S1, self.S2, self.S3)

    def recvData(self, src, d1, d2, d3):

        self.msgsRcvd += 1

        desired = self.x * (src[0] + 1)

        v1 = numpy.fromstring(d1, dtype='float64')
        assert_allclose(v1, desired, atol=1e-07)

        v2 = numpy.array(d2, dtype='float64')
        assert_allclose(v2, desired, atol=1e-07)

        assert_allclose(d3, desired, atol=1e-07)

        if self.msgsRcvd == 3:
            self.msgsRcvd = 0
            self.contribute(None, None, ro.mainProxy.iterationComplete)


charm.start(Main)

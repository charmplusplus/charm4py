import charmpy
from charmpy import charm, Chare, Mainchare, Array, CkNumPes
from charmpy import readonlies as ro
import time
import array
import numpy
from numpy.testing import assert_allclose

charmpy.Options.PROFILING = True
charmpy.Options.LOCAL_MSG_OPTIM = False

MAX_ITER = 10
DATA_LEN = 15000        # number of doubles
CHARES_PER_PE = 10

class Main(Mainchare):

    def __init__(self, args):
        ro.numChares = CkNumPes() * CHARES_PER_PE
        ro.arrayIndexes = [(i,) for i in range(ro.numChares)]
        ro.testProxy = Array(Test, ro.numChares)
        ro.mainProxy = self.thisProxy
        ro.testProxy.doIteration()
        self.iterations = 0
        self.startTime = time.time()

    def iterationComplete(self):
        self.iterations += 1
        print("Iteration", self.iterations, "complete")
        if self.iterations == MAX_ITER:
            print("Program done. Total time =", time.time() - self.startTime)
            charm.printStats()
            charm.exit()
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

    def doIteration(self):
        for idx in ro.arrayIndexes:
            if idx != self.thisIndex:
                self.thisProxy[idx].recvData(self.thisIndex, self.S1, self.S2, self.S3)

    def recvData(self, src, d1, d2, d3):

        self.msgsRcvd += 1

        desired = self.x * (src[0] + 1)

        v1 = numpy.fromstring(d1, dtype='float64')
        assert_allclose(v1, desired, atol=1e-07)

        v2 = numpy.array(d2, dtype='float64')
        assert_allclose(v2, desired, atol=1e-07)

        assert_allclose(d3, desired, atol=1e-07)

        if self.msgsRcvd == (ro.numChares - 1):
            self.msgsRcvd = 0
            self.contribute(None, None, ro.mainProxy.iterationComplete)


charm.start()

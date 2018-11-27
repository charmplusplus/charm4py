from charm4py import charm, Chare, Array, Group, threaded, when, Reducer
from charm4py import readonlies as ro
import time

from charm4py import Options
Options.PROFILING = True

ITERATIONS = 30


class Test(Chare):

    def __init__(self):
        self.iteration = 0
        self.msgsRcvd = 0
        # gather list of PEs on which each array element is located and broadcast to every member
        self.contribute(charm.myPe(), Reducer.gather, self.thisProxy.start)

    @threaded
    def start(self, pes):
        for j in range(ITERATIONS):
            for i in range(ro.numChares):
                x = self.thisProxy[i].getVal(j, ret=True).get()
                assert x == 53 * i * (73 + pes[i]) * j

        self.contribute(None, None, self.thisProxy[0].done)

    @threaded
    @when("self.iteration == iteration")
    def getVal(self, iteration):
        result = 53 * ro.testGroup[charm.myPe()].getVal(ret=True).get() * self.thisIndex[0] * self.iteration
        #assert result == 53 * (73 + charm.myPe()) * self.thisIndex[0] * self.iteration
        self.msgsRcvd += 1
        if self.msgsRcvd == ro.numChares:
            self.msgsRcvd = 0
            self.iteration += 1
        return result

    def done(self):
        charm.printStats()
        exit()


class Test2(Chare):

    def getVal(self): return (73 + charm.myPe())


def main(args):
    ro.numChares = min(charm.numPes() * 8, 32)
    ro.testGroup = Group(Test2)
    Array(Test, ro.numChares)


charm.start(main)

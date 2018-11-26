from charm4py import charm, Chare, Array, Group, threaded, Reducer
from charm4py import readonlies as ro
import time

from charm4py import Options
Options.PROFILING = True

ITERATIONS = 30


class Test(Chare):

    def __init__(self):
        # gather list of PEs on which each array element is located and broadcast to every member
        self.contribute(charm.myPe(), Reducer.gather, self.thisProxy.start)

    @threaded
    def start(self, pes):
        for j in range(ITERATIONS):
            for i in range(ro.numChares):
                x = self.thisProxy[i].getVal(ret=True).get()
                assert x == 53 * i * (73 + pes[i])

        self.contribute(None, None, self.thisProxy[0].done)

    @threaded
    def getVal(self):
        return 53 * ro.testGroup[charm.myPe()].getVal(ret=True).get() * self.thisIndex[0]

    def done(self):
        charm.printStats()
        exit()


class Test2(Chare):

    def getVal(self): return (73 + charm.myPe())


def main(args):
    # every chare sends to every other so don't want a ton of chares
    ro.numChares = min(charm.numPes() * 8, 32)
    ro.testGroup = Group(Test2)
    Array(Test, ro.numChares)


charm.start(main)

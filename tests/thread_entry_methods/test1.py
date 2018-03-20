from charmpy import charm, Chare, Array, Group, threaded
from charmpy import readonlies as ro
import time

from charmpy import Options
Options.PROFILING = True

ITERATIONS = 30


class Test(Chare):

    def __init__(self):
        # gather list of PEs on which each array element is located and broadcast to every member
        self.gather(charm.myPe(), self.thisProxy.start)

    @threaded
    def start(self, pes):
        for j in range(ITERATIONS):
            for i in range(ro.numChares):
                x = self.thisProxy[i].getVal(block=True)
                assert x == 53 * i * (73 + pes[i])

        self.contribute(None, None, self.thisProxy[0].done)

    @threaded
    def getVal(self):
        return 53 * ro.testGroup[charm.myPe()].getVal(block=True) * self.thisIndex[0]

    def done(self):
        charm.printStats()
        charm.exit()


class Test2(Chare):

    def getVal(self): return (73 + charm.myPe())


def main(args):
    ro.numChares = charm.numPes() * 8
    ro.testGroup = Group(Test2)
    Array(Test, ro.numChares)


charm.start(entry=main)

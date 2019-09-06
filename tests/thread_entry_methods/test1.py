from charm4py import charm, Chare, Array, Group, coro, Reducer


charm.options.profiling = True
ITERATIONS = 30


class Test(Chare):

    def __init__(self):
        # gather list of PEs on which each array element is located and broadcast to every member
        self.contribute(charm.myPe(), Reducer.gather, self.thisProxy.start)

    @coro
    def start(self, pes):
        for j in range(ITERATIONS):
            for i in range(numChares):
                x = self.thisProxy[i].getVal(ret=True).get()
                assert x == 53 * i * (73 + pes[i])

        self.reduce(self.thisProxy.verify)

    @coro
    def getVal(self):
        return 53 * testGroup[charm.myPe()].getVal(ret=True).get() * self.thisIndex[0]

    def verify(self):
        assert self._numthreads == 0
        self.reduce(self.thisProxy[0].done)

    def done(self):
        charm.printStats()
        exit()


class Test2(Chare):

    def getVal(self):
        return (73 + charm.myPe())


def main(args):
    global numChares, testGroup
    # every chare sends to every other so don't want a ton of chares
    numChares = min(charm.numPes() * 8, 32)
    testGroup = Group(Test2)
    charm.thisProxy.updateGlobals({'numChares': numChares, 'testGroup': testGroup},
                                  '__main__', awaitable=True).get()
    Array(Test, numChares)


charm.start(main)

from charm4py import charm, Array, Chare, Reducer

CHARES_PER_PE = 4

def main(args):
    testProxy = Array(Test, charm.numPes() * CHARES_PER_PE)

    sum_f = charm.createFuture()
    min_f = charm.createFuture()
    max_f = charm.createFuture()
    testProxy.getStats((sum_f, min_f, max_f))

    print("[Main] Sum: " + str(sum_f.get()) + ", Min: " + str(min_f.get()) + ", Max: " + str(max_f.get()))
    print("[Main] All done.")
    exit()


class Test(Chare):
    def __init__(self):
        pass

    def getStats(self, futures):
        if self.thisIndex[0] == 0:
            self.sum_future, self.min_future, self.max_future = futures

        self.contribute(self.thisIndex[0], Reducer.sum, self.thisProxy[0].collectStats)
        self.contribute(self.thisIndex[0], Reducer.min, self.thisProxy[0].collectStats)
        self.contribute(self.thisIndex[0], Reducer.max, self.thisProxy[0].collectStats)

    def collectStats(self, stat_result):
        assert self.thisIndex[0] == 0, "Reduction target incorrect!"
        if stat_result == 0:
            self.min_future.send(stat_result)
        elif stat_result == (charm.numPes() * CHARES_PER_PE) - 1:
            self.max_future.send(stat_result)
        else:
            self.sum_future.send(stat_result)

charm.start(entry=main)

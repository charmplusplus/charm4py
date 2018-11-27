from charm4py import charm, Chare, Array
from charm4py import readonlies as ro

from charm4py import Options
Options.PROFILING = True

NUM_ITER = 1000
CHARES_PER_PE = 8
MAX_CHARES = 128


class Worker(Chare):

    def sendVal(self):
        ro.main.collectResult(237, self.thisIndex[0] % 2)


class Main(Chare):

    def __init__(self, args):
        ro.main = self.thisProxy
        num_chares = min(charm.numPes() * CHARES_PER_PE, MAX_CHARES)
        assert num_chares % 2 == 0
        workers = Array(Worker, num_chares)
        self.num_responses1 = self.num_responses2 = 0
        self.result = 0
        for i in range(NUM_ITER):
            workers.sendVal()
            self.wait("self.num_responses1 == " + str(num_chares//2))
            self.wait("self.num_responses2 == " + str(num_chares//2))
            assert(self.result == num_chares * 237)
            assert(self.num_responses1 == num_chares//2)
            assert(self.num_responses2 == num_chares//2)
            self.num_responses1 = self.num_responses2 = 0
            self.result = 0
        charm.printStats()
        exit()

    def collectResult(self, result, tag):
        if tag == 0:
            self.num_responses1 += 1
        else:
            self.num_responses2 += 1
        self.result += result


charm.start(Main)

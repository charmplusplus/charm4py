from charm4py import charm, Chare, Array, coro, when, Future


NUM_ITER = 500
CHARES_PER_PE = 8


class Cell(Chare):

    def __init__(self, numChares):
        idx = self.thisIndex[0]
        self.nbs = []
        for i in range(1, 4):
            self.nbs.append(self.thisProxy[(idx + i) % numChares])
            self.nbs.append(self.thisProxy[(idx - i) % numChares])
        self.iteration = -1
        self.msgs_recvd = 0

    @coro
    def work(self, done_fut):
        for self.iteration in range(NUM_ITER):
            for nb in self.nbs:
                nb.recvData(self.iteration, None)
            self.iter_complete = Future()
            self.iter_complete.get()
        self.reduce(done_fut)

    @when('self.iteration == iteration')
    def recvData(self, iteration, data):
        self.msgs_recvd += 1
        if self.msgs_recvd == len(self.nbs):
            self.msgs_recvd = 0
            self.iter_complete()


def main(args):
    numChares = charm.numPes() * CHARES_PER_PE
    cells = Array(Cell, numChares, args=[numChares])
    charm.awaitCreation(cells)
    f = Future()
    cells.work(f)
    f.get()
    exit()


charm.start(main)

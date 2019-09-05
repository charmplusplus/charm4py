from charm4py import charm, Chare, Group, coro, Reducer, Future


NUM_ITER = 100


class Test(Chare):

    def __init__(self):
        self.numallreduce = 0

    @coro
    def work(self, f, numchares, secproxy=None):
        for _ in range(NUM_ITER):
            result = self.allreduce(1, Reducer.sum, section=secproxy).get()
            assert result == numchares
            self.numallreduce += 1
        if self.thisIndex == 0:
            f()

    def verify(self):
        if self.thisIndex % 2 == 0:
            assert self.numallreduce == 2 * NUM_ITER
        else:
            assert self.numallreduce == NUM_ITER


def main(args):
    assert charm.numPes() % 2 == 0
    g = Group(Test)
    gsec = g[::2]  # make a section with even numbered elements

    f = Future(2)
    g.work(f, charm.numPes())
    gsec.work(f, charm.numPes() // 2, gsec)
    f.get()
    g.verify(awaitable=True).get()
    exit()


charm.start(main)

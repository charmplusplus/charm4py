from charm4py import charm, Chare, Group, Array, coro, Future


NUM_ITER = 300


class Test(Chare):

    @coro
    def start(self, done_future):
        self.done_future = done_future
        self.iteration = 0
        for _ in range(NUM_ITER):
            assert self.thisProxy[self.thisIndex].work(ret=True).get() == 3625
        self.reduce(self.thisProxy.verify)

    @coro
    def work(self):
        if self.iteration % 2 == 0:
            mype = charm.myPe()
            assert charm.thisProxy[mype].myPe(ret=True).get() == mype
        self.iteration += 1
        return 3625

    def verify(self):
        assert self._numthreads == 0
        self.reduce(self.done_future)


def main(args):
    numChares = charm.numPes() * 10
    a = Array(Test, numChares)
    g = Group(Test)
    charm.awaitCreation(a, g)
    f1 = Future()
    f2 = Future()
    a.start(f1)
    g.start(f2)
    f1.get()
    f2.get()
    exit()


charm.start(main)

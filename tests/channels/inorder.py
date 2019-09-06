from charm4py import charm, Chare, Group, coro, Future, Channel


NUM_STEPS = 5000


class Test(Chare):

    @coro
    def work(self, mainProxy, done_fut):
        ch = Channel(self, remote=mainProxy)
        for i in range(NUM_STEPS):
            assert ch.recv() == i
        done_fut()


class Main(Chare):

    def __init__(self, args):
        assert charm.numPes() >= 2
        g = Group(Test)
        done_fut = Future()
        g[1].work(self.thisProxy, done_fut)
        ch = Channel(self, remote=g[1])
        for i in range(NUM_STEPS):
            ch.send(i)
        done_fut.get()
        exit()


charm.start(Main)

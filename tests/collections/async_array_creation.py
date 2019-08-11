from charm4py import charm, Chare, Reducer, coro, Future


class Test(Chare):

    def __init__(self, v=0):
        self.val = v

    def work(self, cb, x=0):
        self.contribute(x + self.val, Reducer.sum, cb)

    def work2(self, cb):
        self.contribute(charm.myPe(), Reducer.sum, cb)


class Controller(Chare):

    @coro
    def start(self):
        assert charm.myPe() == 1

        N = charm.numPes() * 3
        a1 = charm.thisProxy[0].createArray(Test, N, ret=True).get()
        f = Future()
        a1.work(f, 5)
        assert f.get() == N * 5

        N = 25
        a2 = charm.thisProxy[0].createArray(Test, (5, 5), args=[33], ret=True).get()
        f = Future()
        a2.work(f, 6)
        assert f.get() == N * (6 + 33)

        exit()


def main(args):
    controller = Chare(Controller, onPE=1)
    controller.start()


charm.start(main)

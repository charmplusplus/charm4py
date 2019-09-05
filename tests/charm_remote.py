from charm4py import charm, Chare, Group, coro
import random

charm.options.remote_exec = True


class Controller(Chare):

    @coro
    def start(self):
        # print('Controller running on PE', charm.myPe())
        for i in range(charm.numPes()):
            assert i == charm.thisProxy[i].myPe(ret=True).get()

        pe = charm.myPe() - 1
        if pe == -1:
            pe = 0
        charm.thisProxy[pe].exec('global MY_GLOBAL; MY_GLOBAL = 7262', __name__, awaitable=True).get()
        assert charm.thisProxy[pe].eval('MY_GLOBAL', __name__, ret=True).get() == 7262

        Group(Test)


class Test(Chare):

    def __init__(self):
        self.contribute(None, None, charm.thisProxy[0].exit)


def main(args):
    if charm.numPes() > 1:
        pes = list(range(1, charm.numPes()))
    else:
        pes = [0]
    Chare(Controller, onPE=random.choice(pes)).start()


charm.start(main)

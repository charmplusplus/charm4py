from charm4py import charm, Chare, Group, threaded
import random


class Controller(Chare):

    @threaded
    def start(self):
        # print('Controller running on PE', charm.myPe())
        for i in range(charm.numPes()):
            assert i == charm.thisProxy[i].myPe(ret=True).get()

        g = Group(Test)


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

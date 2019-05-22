from charm4py import charm, Chare, Group, Array, threaded


class Test(Chare):

    @threaded
    def __init__(self, x):
        if charm.myPe() == 0:
            print(self.thisIndex, x)


def main(args):
    g = Group(Test, args=[33])
    a = Array(Test, charm.numPes() * 4, args=[50])
    charm.awaitCreation(g, a)
    exit()


charm.start(main)

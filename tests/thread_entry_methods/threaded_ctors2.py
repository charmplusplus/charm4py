from charm4py import charm, Chare, Group, Array, threaded


class A(Chare):

    @threaded
    def __init__(self):
        self.x = 65824

    @threaded
    def getVal(self):
        return self.x


class B(Chare):

    @threaded
    def __init__(self, grp_proxy):
        x = grp_proxy[self.thisIndex[0]].getVal(ret=True).get()
        assert x == 65824


def main(args):
    g = Group(A)
    a = Array(B, charm.numPes(), args=[g])
    charm.awaitCreation(a)
    exit()


charm.start(main)

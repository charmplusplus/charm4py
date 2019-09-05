from charm4py import charm, Chare, Group, Array


class Test(Chare):

    def getProxy(self, elem=False):
        if elem:
            return self.thisProxy[self.thisIndex]
        else:
            return self.thisProxy


class Main(Chare):

    def __init__(self, args):
        assert charm.numPes() >= 4
        g1 = Group(Test)
        g2 = Group(Test)
        a = Array(Test, charm.numPes() * 8)

        assert self.thisProxy == self.thisProxy
        assert self.thisProxy != g1
        assert self.thisProxy != a

        assert g1 == g1
        assert g1 == g1[2].getProxy(ret=True).get()
        assert g1[2] == g1[2].getProxy(elem=True, ret=True).get()
        assert g1[2].getProxy(ret=True).get() == g1[3].getProxy(ret=True).get()
        assert g1[2].getProxy(True, ret=True).get() != g1[3].getProxy(True, ret=True).get()

        assert g1 != g2
        assert g1[2].getProxy(ret=True).get() != g2[2].getProxy(ret=True).get()
        assert g1[2].getProxy(True, ret=True).get() != g2[2].getProxy(True, ret=True).get()

        assert g1 != a
        assert a == a
        assert a == a[12].getProxy(ret=True).get()
        assert a[12] == a[12].getProxy(elem=True, ret=True).get()
        assert a[8] != a[12].getProxy(elem=True, ret=True).get()

        exit()


charm.start(Main)

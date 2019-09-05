from charm4py import charm, Chare, Group, Array, coro


class Test(Chare):

    def getIdx(self):
        return self.thisIndex

    @coro
    def getIdx_th(self):
        return self.thisIndex


def main(args):

    g = Group(Test)
    elems = list(range(0, charm.numPes(), 2))
    assert g[::2].getIdx(ret=True).get() == elems
    assert g[0::2].getIdx_th(ret=True).get() == elems
    assert g[:charm.numPes():2].getIdx(ret=True).get() == elems
    assert g[0:charm.numPes()].getIdx_th(ret=True).get() != elems

    a1 = Array(Test, (8, 8))
    a2 = Array(Test, 64)

    indexes = a1[0:8:2, 1:8:2].getIdx(ret=True).get()
    assert len(indexes) == 8*8//4
    for idx in indexes:
        assert len(idx) == 2
        assert idx[0] % 2 == 0
        assert idx[1] % 2 != 0

    indexes = a2[0:64:5].getIdx_th(ret=True).get()
    assert len(indexes) == 13
    for idx in indexes:
        assert len(idx) == 1
        assert idx[0] % 5 == 0

    exit()


charm.start(main)

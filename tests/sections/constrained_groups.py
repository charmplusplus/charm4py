from charm4py import charm, Chare, Group, Reducer, coro
import random


class Test(Chare):

    def __init__(self, x):
        assert x == 4862
        assert self.thisIndex in section_pes

    def test(self):
        assert self.thisIndex in section_pes

    def test2(self):
        return 34589

    def getIdx(self):
        return self.thisIndex

    @coro
    def testallreduce(self):
        result = self.allreduce(self.thisIndex, Reducer.gather).get()
        assert result == sorted(section_pes)


def main(args):
    assert charm.numPes() > 1
    section_pes = random.sample(range(charm.numPes()), charm.numPes() // 2)
    charm.thisProxy.updateGlobals({'section_pes': section_pes}, ret=1).get()
    g = Group(Test, onPEs=section_pes, args=[4862])
    assert g[section_pes[0]].test2(ret=1).get() == 34589
    g.test(ret=1).get()

    assert g.getIdx(ret=2).get() == sorted(section_pes)
    assert g[section_pes[0]].getIdx(ret=1).get() == section_pes[0]

    g.testallreduce(ret=1).get()

    exit()


charm.start(main)

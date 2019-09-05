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
    global section_pes
    section_pes = random.sample(range(charm.numPes()), charm.numPes() // 2)
    charm.thisProxy.updateGlobals({'section_pes': section_pes}, awaitable=True).get()
    g = Group(Test, onPEs=section_pes, args=[4862])
    assert g[section_pes[0]].test2(ret=True).get() == 34589
    g.test(awaitable=True).get()

    assert g.getIdx(ret=True).get() == sorted(section_pes)
    assert g[section_pes[0]].getIdx(ret=True).get() == section_pes[0]

    g.testallreduce(awaitable=True).get()

    exit()


charm.start(main)

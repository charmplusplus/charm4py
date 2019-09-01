from charm4py import charm, Chare, Group, coro
import proxies_same_name_aux


class Hello(Chare):

    def check1(self):
        assert charm.myPe() == 3
        return 68425


class Test(Chare):

    def __init__(self):
        assert charm.myPe() == 2

    @coro
    def test(self, proxy, method_name):
        assert getattr(proxy[1], method_name)(ret=True).get() == 32255


def main(args):
    assert charm.numPes() >= 4
    g1 = Group(Hello)
    g2 = Group(proxies_same_name_aux.Hello)
    tester1 = Chare(Test, onPE=2)
    tester2 = Chare(proxies_same_name_aux.Test, onPE=1)
    charm.awaitCreation(g2, g1, tester2, tester1)
    tester1.test(g2, 'check2', awaitable=True).get()
    tester2.test(g1, 'check1', awaitable=True).get()
    exit()


charm.start(main, modules=['proxies_same_name_aux'])

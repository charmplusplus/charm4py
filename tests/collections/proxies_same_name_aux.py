from charm4py import charm, Chare, threaded


class Hello(Chare):

    def check2(self):
        assert charm.myPe() == 1
        return 32255


class Test(Chare):

    def __init__(self):
        assert charm.myPe() == 1

    @threaded
    def test(self, proxy, method_name):
        assert getattr(proxy[3], method_name)(ret=1).get() == 68425

from charm4py import charm, Chare, Array


class Test(Chare):
    def __init__(self):
        pass


class Main(Chare):

    def __init__(self, args):
        a = Array(Test, charm.numPes() * 8)

        # destroy element 0
        a[0].ckDestroy()

        assert not (0 in charm.arrays[a.aid]), "Index not destroyed"

        charm.exit()


charm.start(Main)

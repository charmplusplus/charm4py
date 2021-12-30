from charm4py import charm, Chare, Group, Array, Reducer


g = None


class TestArray(Chare):
    def increment_count(self):
        obj = g.ckLocalBranch()
        obj.increment_count()


class TestGroup(Chare):
    def __init__(self):
        self.local_count = 0

    def increment_count(self):
        self.local_count += 1
        if self.local_count == 8:
            self.reduce(self.thisProxy[0].done, 1, Reducer.sum)

    def done(self, result):
        charm.exit()


class Main(Chare):

    def __init__(self, args):
        g = Group(TestGroup)
        a = Array(TestArray, charm.numPes() * 8)

        charm.thisProxy.updateGlobals({'g': g}, awaitable=True).get()

        a.increment_count()


charm.start(Main)

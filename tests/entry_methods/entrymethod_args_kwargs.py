from charm4py import charm, Chare, Group, Array


class Test(Chare):

    def startPhase(self, x, y, a, b):
        self.expected_x = x
        self.expected_y = y
        self.expected_a = a
        self.expected_b = b

    def recv(self, x, y, a=33, b=44):
        assert x == self.expected_x
        assert y == self.expected_y
        assert a == self.expected_a
        assert b == self.expected_b


class Main(Chare):

    def startPhase(self, x, y, a, b):
        self.expected_x = x
        self.expected_y = y
        self.expected_a = a
        self.expected_b = b

    def recv(self, x, y, a=33, b=44):
        assert x == self.expected_x
        assert y == self.expected_y
        assert a == self.expected_a
        assert b == self.expected_b

    def __init__(self, args):
        assert charm.numPes() >= 2

        g = Group(Test)
        a = Array(Test, charm.numPes() * 4)

        self.thisProxy.startPhase(1, 2, 33, 44, awaitable=True).get()
        g.startPhase(1, 2, 33, 44, awaitable=True).get()
        a.startPhase(1, 2, 33, 44, awaitable=True).get()
        for collection in (g, a, self.thisProxy):
            collection.recv(1, 2, awaitable=True).get()
            collection.recv(1, 2, 33, awaitable=True).get()
            collection.recv(1, 2, 33, 44, awaitable=True).get()
            collection.recv(y=2, x=1, awaitable=True).get()
            collection.recv(b=44, a=33, y=2, x=1, awaitable=True).get()
            if collection == g:
                single_chare = 1
            elif collection == a:
                single_chare = 4
            else:
                continue
            collection[single_chare].recv(1, 2, awaitable=True).get()
            collection[single_chare].recv(1, 2, 33, awaitable=True).get()
            collection[single_chare].recv(1, 2, 33, 44, awaitable=True).get()
            collection[single_chare].recv(y=2, x=1, awaitable=True).get()
            collection[single_chare].recv(b=44, a=33, y=2, x=1, awaitable=True).get()

        self.thisProxy.startPhase(10, 20, 3000, 4000, awaitable=True).get()
        g.startPhase(10, 20, 3000, 4000, awaitable=True).get()
        a.startPhase(10, 20, 3000, 4000, awaitable=True).get()
        for collection in (g, a, self.thisProxy):
            collection.recv(10, 20, 3000, 4000, awaitable=True).get()
            collection.recv(10, 20, 3000, b=4000, awaitable=True).get()
            collection.recv(b=4000, a=3000, y=20, x=10, awaitable=True).get()
            if collection == g:
                single_chare = 1
            elif collection == a:
                single_chare = 4
            else:
                continue
            collection[single_chare].recv(10, 20, 3000, b=4000, awaitable=True).get()
            collection[single_chare].recv(b=4000, a=3000, y=20, x=10, awaitable=True).get()

        exit()


charm.start(Main)

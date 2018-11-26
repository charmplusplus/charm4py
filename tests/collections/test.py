from charm4py import charm, Chare, Group, Array, Reducer

# Test same Chare class (Test) used in multiple collection types (Group, Array)

CHARES_PER_PE = 5


class Main(Chare):
    def __init__(self, args):
        Group(Test).work(self.thisProxy)
        Array(Test, charm.numPes() * CHARES_PER_PE).work(self.thisProxy)
        self.countReductions = self.count = 0

    def done(self, result):
        self.count += result
        self.countReductions += 1
        if self.countReductions == 2:
          assert self.count == (charm.numPes() + charm.numPes() * CHARES_PER_PE)
          print("Program done")
          exit()


class Test(Chare):

    def __init__(self):
        if isinstance(self.thisIndex, tuple): myIndex = self.thisIndex[0]
        else: myIndex = self.thisIndex
        if charm.numPes() <= 20 or myIndex == 0:
          print("Test", self.thisIndex, "created")

    def work(self, main):
        self.contribute(1, Reducer.sum, main.done)


charm.start(Main)

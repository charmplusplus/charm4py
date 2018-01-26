from charmpy import charm, Mainchare, Chare, Group, Array, Reducer
import charmpy

# Test same Chare class (Test) used in multiple collection types (Group, Array)

class Main(Mainchare):
    def __init__(self, args):
        Group(Test).work(self.thisProxy)
        Array(Test, 20).work(self.thisProxy)
        self.countReductions = self.count = 0

    def done(self, result):
        self.count += result
        self.countReductions += 1
        if self.countReductions == 2:
          assert self.count == (charm.numPes() + 20)
          print("Program done")
          charm.exit()

class Test(Chare):

    def __init__(self):
        print("Test", self.thisIndex, "created")

    def work(self, main):
        self.contribute(1, Reducer.sum, main.done)

charm.start()

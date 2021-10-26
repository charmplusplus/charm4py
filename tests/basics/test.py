from charm4py import charm, Chare, Array

class Elem(Chare):
    def __init__(self):
        self.myIndex = self.thisIndex[0]
        #print("PE{}".format(self.myPe()))
        if self.myIndex == 0:
            self.neighbor = self.thisProxy[1]
            self.data = list(range(100))
            self.neighbor.recv(self.data)
        else:
            self.neighbor = self.thisProxy[0]

    def recv(self, data):
        data[0] = 100
        print("PE{} :: data[0] = {}".format(self.myIndex, data[0]))
        self.neighbor.done()

    def done(self):
        print("PE{} :: data[0] = {}".format(self.myIndex, self.data[0]))
        assert(self.data[0] == 0)
        exit()

def main(args):
    elems = Array(Elem, 2)
    charm.awaitCreation(elems)



charm.start(main)

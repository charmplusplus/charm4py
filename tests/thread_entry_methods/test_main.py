from charm4py import charm, Chare, Group

class Test(Chare):

    def getData(self):
        return self.thisIndex**2


def main(args):
    grpProxy = Group(Test)
    for i in range(charm.numPes()):
        data = grpProxy[i].getData(ret=True).get()
        assert data == i**2, "Blocking call in main failed."
        print("Test " + str(i) + " sent data: " + str(data))
    exit()

charm.start(main)

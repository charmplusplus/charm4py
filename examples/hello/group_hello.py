from charm4py import charm, Chare, Group


class Hello(Chare):

    def __init__(self):
        print("Hello " + str(self.thisIndex) + " created")

    def SayHi(self, hiNo):
        print("Hi[" + str(hiNo) + "] from element " + str(self.thisIndex))
        if self.thisIndex + 1 < charm.numPes():
            # Pass the hello on:
            self.thisProxy[self.thisIndex+1].SayHi(hiNo+1)
        else:
            print("All done")
            exit()


def main(args):
    print("Running Hello on " + str(charm.numPes()) + " processors")
    Group(Hello)[0].SayHi(17)


charm.start(main)

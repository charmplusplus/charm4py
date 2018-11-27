from charm4py import charm, Chare, Group


class HelloList:
    def __init__(self, hiNo):
        self.hiNo = hiNo
        self.hellos = []
    def addHello(self, msg):
        self.hellos.append(msg)
    def __str__(self):
        return "RESULT:\n" + "\n".join(self.hellos)


class Hello(Chare):

    def __init__(self):
        print("Hello " + str(self.thisIndex) + " created")

    def SayHi(self, hellos):
        hellos.addHello("Hi[" + str(hellos.hiNo) + "] from element " + str(self.thisIndex))
        hellos.hiNo += 1
        if self.thisIndex + 1 < charm.numPes():
            # Pass the hello list on:
            self.thisProxy[self.thisIndex+1].SayHi(hellos)
        else:
            print("All done " + str(hellos))
            exit()


def main(args):
    print("Running Hello on " + str(charm.numPes()) + " processors")
    Group(Hello)[0].SayHi(HelloList(17))


charm.start(main)

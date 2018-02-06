from charmpy import Chare, Mainchare, Group, charm

class Main(Mainchare):

    def __init__(self, args):
        # create Group of Hello objects (one object exists and runs on each core)
        hellos = Group(Hello)
        # call method 'SayHello' of all group members, passing proxy to myself
        hellos.SayHi(self.thisProxy)

    # called when every element has contributed
    def done(self):
        charm.exit()

class Hello(Chare):

    def __init__(self):
        pass

    def SayHi(self, main):
        print("Hello World from element", self.thisIndex)
        # contribute to empty reduction to end program
        self.contribute(None, None, main.done)

charm.start()

# examples/tutorial/hello_world2.py
from charm4py import Chare, Group, charm

class Hello(Chare):

    def SayHi(self, future):
        print("Hello World from element", self.thisIndex)
        self.contribute(None, None, future)

def main(args):
    # create Group of Hello objects (one object exists and runs on each core)
    hellos = Group(Hello)
    # call method 'SayHi' of all group members, wait for method to be invoked on all
    f = charm.createFuture()
    hellos.SayHi(f)
    f.get()
    exit()

charm.start(main)

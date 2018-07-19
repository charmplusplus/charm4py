# examples/tutorial/hello_world.py
from charmpy import Chare, Group, charm

class Hello(Chare):

    def SayHi(self):
        print("Hello World from element", self.thisIndex)

def main(args):
    # create Group of Hello objects (one object exists and runs on each core)
    hellos = Group(Hello)
    # call method 'SayHi' of all group members, wait for method to be invoked on all
    hellos.SayHi(ret=True).get()
    charm.exit()

charm.start(main)


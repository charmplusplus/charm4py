from charmpy import charm, Chare
from charmpy import readonlies as ro
import time

class Hello(Chare):

    def SayHi(self):
        print("Hello from PE", charm.myPe(), "on", time.strftime('%c'))
        ro.byes[(self.thisIndex + 1) % charm.numPes()].SayGoodbye()

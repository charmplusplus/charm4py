from charm4py import charm, Chare
from charm4py import readonlies as ro
import time

class Hello(Chare):

    def SayHi(self):
        print("Hello from PE", charm.myPe(), "on", time.strftime('%c'))
        ro.byes[(self.thisIndex + 1) % charm.numPes()].SayGoodbye()

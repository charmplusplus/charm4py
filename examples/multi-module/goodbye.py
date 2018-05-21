from charmpy import charm, Chare
from charmpy import readonlies as ro
import time

class Goodbye(Chare):

    def SayGoodbye(self):
        print("Goodbye from PE", charm.myPe())
        self.contribute(None, None, ro.mainProxy.done)

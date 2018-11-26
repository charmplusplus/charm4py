from charm4py import charm, Chare
from charm4py import readonlies as ro
import time

class Goodbye(Chare):

    def SayGoodbye(self):
        print("Goodbye from PE", charm.myPe())
        self.contribute(None, None, ro.mainProxy.done)

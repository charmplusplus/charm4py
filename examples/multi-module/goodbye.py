from charmpy import Chare, CkMyPe
from charmpy import readonlies as ro
import time

class Goodbye(Chare):
    def __init__(self):
        pass

    def SayGoodbye(self):
        print("Goodbye from PE", CkMyPe())
        self.contribute(None, None, ro.mainProxy.done)

from charmpy import Chare, CkMyPe, CkNumPes
from charmpy import readonlies as ro
import time

class Hello(Chare):
    def __init__(self):
        pass

    def SayHi(self):
        print("Hello from PE", CkMyPe(), "on", time.strftime('%c'))
        ro.byes[(self.thisIndex + 1) % CkNumPes()].SayGoodbye()

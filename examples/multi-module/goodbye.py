from charmpy import Group, CkMyPe
from charmpy import readonlies as ro
import time

class Goodbye(Group):
    def __init__(self):
        pass

    def SayGoodbye(self):
        print("Goodbye from PE", CkMyPe())
        self.contribute(None, None, ro.mainProxy.done)

from charmpy import Group, CkMyPe, CkNumPes
from charmpy import readonlies as ro
import time

class Hello(Group):
    def __init__(self):
        super(Hello, self).__init__()

    def SayHi(self):
        print("Hello from PE", CkMyPe(), "on", time.strftime('%c'))
        ro.byeProxy[(self.thisIndex + 1) % CkNumPes()].SayGoodbye()

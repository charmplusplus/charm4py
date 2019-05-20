from charm4py import charm, Chare
import time


class Hello(Chare):

    def SayHi(self):
        print('Hello from PE', charm.myPe(), 'on', time.strftime('%c'))
        byes[(self.thisIndex + 1) % charm.numPes()].SayGoodbye()

from charm4py import charm, Chare
import time


class Goodbye(Chare):

    def SayGoodbye(self):
        print('Goodbye from PE', charm.myPe())
        self.reduce(mainProxy.done)

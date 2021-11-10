from charm4py import charm, Chare, Array
from time import sleep

class Elem(Chare):
    def __init__ (self, main_proxy):
        self.main_proxy = main_proxy

    def recv(self, data):
        data[0] = 100
        print("Chare :: data[0] = {}".format(data[0]))
        self.main_proxy.done()

class Main(Chare):
    def __init__(self):
        self.data = list(range(10))
        elem = Chare(Elem, args=[self.thisProxy])
        charm.awaitCreation(elem)
        elem.recv(self.data)

    def done(self):
        assert(self.data[0] == 0)
        exit()





def main(args):
    m = Chare(Main)
    charm.awaitCreation(m)

charm.start(main)

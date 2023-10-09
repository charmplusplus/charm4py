from charm4py import charm, Chare, Array, ray
from time import sleep

class MyChare(Chare):

    def __init__(self):
        print('Hello from MyChare instance in processor', charm.myPe(), 'index', self.thisIndex)

    def add(self, a, b):
        sleep(2)
        print("ADD", a, b)
        return a + b


def main(args):
    ray.init()
    # create 3 instances of MyChare, distributed among cores by the runtime
    arr = Array(MyChare, 3)

    c = arr[0].add(1, 2)
    d = arr[1].add(c, 3)

    #sleep(5)
    #exit()


charm.start(main)

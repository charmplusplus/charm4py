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

    c = arr[1].add(1, 2)
    d = arr[2].add(c, 3)
    e = arr[2].add(4, 5)

    #sleep(5)
    #exit()


charm.start(main)

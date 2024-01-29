from charm4py import charm, Chare, Array, ray
from time import sleep

@ray.remote
class Compute(object):
    def __init__(self, arg):
        print('Hello from MyChare instance in processor', charm.myPe(), 'index', self.thisIndex, arg)

    def add(self, a, b):
        sleep(2)
        print("ADD", a, b)
        return a + b


def main(args):
    ray.init()
    # create 3 instances of MyChare, distributed among cores by the runtime
    arr = [Compute.remote(i) for i in range(4)]

    c = arr[0].add(1, 2) # fut id 0
    d = arr[1].add(3, c) # fut id 1
    e = arr[2].add(2, d)
    f = arr[3].add(c, 4)

    print(c.get())
    
    not_ready = [c, d, e, f]
    while len(not_ready) > 0:
        ready, not_ready = ray.wait(not_ready)
        print("Fetched value: ", ray.get(ready))

    #sleep(10)
    #exit()


charm.start(main)

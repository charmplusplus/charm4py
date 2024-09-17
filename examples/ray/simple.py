from charm4py import charm, coro, Chare, Array, ray
from time import sleep
import numpy as np


@ray.remote
def add_task(a, b):
    sleep(2)
    if a != -1:
        res = add_task.remote(-1, a)
        res = ray.get(res)
    print("Add task", a, b)
    return a + b

@ray.remote
class Compute(object):
    def __init__(self, arg):
        print('Hello from MyChare instance in processor', charm.myPe(), 'index', self.thisIndex, arg)

    def add(self, a, b):
        sleep(2)
        print("Add actor method", a, b)
        return a + b


def main(args):
    ray.init()
    # create 3 instances of MyChare, distributed among cores by the runtime
    arr = [Compute.remote(i) for i in range(4)]
    
    obj1 = np.arange(100)
    obj2 = np.arange(100)
    a = ray.put(obj1)
    b = ray.put(obj2)
    c = arr[0].add.remote(1, 2) # fut id 0
    d = arr[1].add.remote(3, c) # fut id 1
    e = arr[2].add.remote(2, d)
    f = arr[3].add.remote(c, 4)
    g = arr[3].add.remote(a, b)
    h = add_task.remote(e, f)

    not_ready = [c, d, e, f, g, h]
    while len(not_ready) > 0:
        ready, not_ready = ray.wait(not_ready)
        print("Fetched value: ", ray.get(ready))

    exit()


charm.start(main)

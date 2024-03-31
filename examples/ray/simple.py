from charm4py import charm, coro, Chare, Array, ray
from time import sleep


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

    c = arr[0].add(1, 2) # fut id 0
    d = arr[1].add(3, c) # fut id 1
    e = arr[2].add(2, d)
    f = arr[3].add(c, 4)
    g = add_task.remote(e, f)

    not_ready = [c, d, e, f, g]
    while len(not_ready) > 0:
        ready, not_ready = ray.wait(not_ready)
        print("Fetched value: ", ray.get(ready))

    exit()


charm.start(main)

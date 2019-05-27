from charm4py import charm, Chare, Array
from time import time


NUM_CHARES = 4


class Test(Chare):

    def __init__(self):
        self.t0 = time()

    def start(self):
        charm.scheduleCallableAfter(self.thisProxy[self.thisIndex].next, 1, [-1])

    def next(self, from_elem):
        print(self.thisIndex, 'time=', time() - self.t0, 'from=', from_elem)
        assert from_elem == self.thisIndex[0] - 1
        assert time() - self.t0 > self.thisIndex[0] + 0.9
        if self.thisIndex[0] == NUM_CHARES - 1:
            print('DONE')
            exit()
        else:
            charm.scheduleCallableAfter(self.thisProxy[self.thisIndex[0] + 1].next,
                                        1, [self.thisIndex[0]])


def main(args):
    a = Array(Test, NUM_CHARES)
    a[0].start()


charm.start(main)

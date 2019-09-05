from charm4py import charm, Chare, Array, Reducer, Future
import random


def member(obj):
    if obj.thisIndex[1] % 2 == 1:
        return 0
    else:
        return -1


class Test(Chare):

    def __init__(self):
        self.insection = (member(self) >= 0)

    def setSecProxy(self, proxy):
        self.secProxy = proxy

    def ping(self, f, check=True):
        assert self.insection
        self.contribute(self.thisIndex, Reducer.gather, f, self.secProxy)

    def getElems(self):
        return self.thisIndex

    def ping2(self, f, secProxy):
        self.contribute(self.thisIndex, Reducer.gather, f, secProxy)


def main(args):
    array2d = Array(Test, (8, 8))
    array3d = Array(Test, (4, 5, 3))

    # for each array, create one section using member function to determine section membership
    for array, size in [(array2d, 8*8), (array3d, 4*5*3)]:
        secProxy = charm.split(array, 1, member)[0]
        array.setSecProxy(secProxy, awaitable=True).get()
        f = Future()
        secProxy.ping(f)
        assert len(f.get()) < size

    # for each array, create one section passing a random list of element indexes (half the size of the array)
    for array, size in [(array2d, 8*8), (array3d, 4*5*3)]:
        elems = array.getElems(ret=True).get()
        assert len(elems) == size
        section_elems = random.sample(elems, size // 2)
        secProxy = charm.split(array, 1, elems=section_elems)[0]
        f = Future()
        secProxy.ping2(f, secProxy)
        assert f.get() == sorted(section_elems)
        assert secProxy.getElems(ret=True).get() == sorted(section_elems)

    exit()


charm.start(main)

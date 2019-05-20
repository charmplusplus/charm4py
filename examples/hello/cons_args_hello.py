from charm4py import charm, Chare, Array, Group
import numpy


class Hello(Chare):

    def __init__(self, arg0, arg1, arg2):
        assert [arg0, arg1, arg2] == [1, 'test', [4.2, -9]], 'Construtor args for array failed.'
        print('Hello', self.thisIndex, 'created on PE', charm.myPe())

    def SayHi(self, hiNo):
        print('Hi[' + str(hiNo) + '] from element', self.thisIndex, 'on PE', charm.myPe())
        if self.thisIndex == lastIdx:
            print('Array done')
            grpProxy[0].SayHi(17)
        else:
            nextIndex = list(self.thisIndex)
            for i in range(nDims-1,-1,-1):
                nextIndex[i] = (nextIndex[i] + 1) % ARRAY_SIZE[i]
                if nextIndex[i] != 0: break
            return self.thisProxy[nextIndex].SayHi(hiNo+1)


class HelloGroup(Chare):

    def __init__(self, arg0, arg1, arg2, arg3):
        assert [arg0, arg1, arg2] == [1, 'test', [4.2, -9]], 'Constructor args for groups failed.'
        assert arg3.all() == numpy.full((3,5), 4.2).all(), 'Numpy constructor arg for groups failed.'
        print('HelloGroup', self.thisIndex, 'created')

    def SayHi(self, hiNo):
        print('Hi[' + str(hiNo) + '] from element', self.thisIndex)
        if self.thisIndex + 1 < charm.numPes():
            # Pass the hello on:
            self.thisProxy[self.thisIndex+1].SayHi(hiNo+1)
        else:
            print('All done')
            exit()


def main(args):

    if len(args) <= 1:
        args = [None, 3, 2]  # default: 3 dimensions of size 2 each
    elif len(args) != 3:
        charm.abort('Usage : python array_hello.py [<num_dimensions> <array_size>]')

    nDims = int(args[1])
    ARRAY_SIZE = [int(args[2])] * nDims
    firstIdx = [0] * nDims
    lastIdx = tuple([x-1 for x in ARRAY_SIZE])
    myglobals = {'nDims': nDims, 'ARRAY_SIZE': ARRAY_SIZE, 'firstIdx': firstIdx, 'lastIdx': lastIdx}
    charm.thisProxy.updateGlobals(myglobals, module_name='__main__', ret=True).get()

    nElements = 1
    for x in ARRAY_SIZE:
        nElements *= x
    print('Running Hello on', charm.numPes(), 'processors for', nElements, 'elements, array dims=', ARRAY_SIZE)
    arrProxy = Array(Hello, ARRAY_SIZE, args=[1, 'test', [4.2, -9]])
    grpProxy = Group(HelloGroup, args=[1, 'test', [4.2, -9], numpy.full((3,5), 4.2)])

    charm.thisProxy.updateGlobals({'grpProxy': grpProxy}, ret=True).get()
    arrProxy[firstIdx].SayHi(17)


charm.start(main)

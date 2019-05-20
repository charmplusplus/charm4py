from charm4py import charm, Chare, Array


class Hello(Chare):

    def __init__(self):
        print('Hello', self.thisIndex, 'created on PE', charm.myPe())

    def SayHi(self, hiNo):
        print('Hi[' + str(hiNo) + '] from element', self.thisIndex, 'on PE', charm.myPe())
        if self.thisIndex == lastIdx:
            print('All done')
            exit()
        else:
            nextIndex = list(self.thisIndex)
            for i in range(nDims-1,-1,-1):
                nextIndex[i] = (nextIndex[i] + 1) % ARRAY_SIZE[i]
                if nextIndex[i] != 0:
                    break
            return self.thisProxy[nextIndex].SayHi(hiNo+1)


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
    print('Running Hello on', charm.numPes(), 'processors for', nElements,
          'elements, array dims=', ARRAY_SIZE)
    arrProxy = Array(Hello, ARRAY_SIZE)
    arrProxy[firstIdx].SayHi(17)


charm.start(main)

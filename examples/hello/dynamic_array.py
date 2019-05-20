from charm4py import charm, Chare, Array, Reducer
import itertools


class Main(Chare):

    def __init__(self, args):

        if len(args) <= 1:
            args = [None,2,3]  # default: 2 dimensions of size 3
        elif len(args) != 3:
            charm.abort('Usage : python array_hello.py [<num_dimensions> <array_size>]')

        nDims = int(args[1])
        ARRAY_SIZE = [int(args[2])] * nDims
        firstIdx = [0] * nDims
        lastIdx = tuple([x-1 for x in ARRAY_SIZE])
        myglobals = {'nDims': nDims, 'ARRAY_SIZE': ARRAY_SIZE, 'firstIdx': firstIdx,
                     'lastIdx': lastIdx, 'mainProxy': self.thisProxy}
        charm.thisProxy.updateGlobals(myglobals, module_name='__main__', ret=True).get()

        self.nElements = 1
        for x in ARRAY_SIZE:
            self.nElements *= x
        print('Running Hello on', charm.numPes(), 'processors for', self.nElements, 'elements')
        self.arrProxy = Array(Hello, ndims=nDims)
        print('Created array proxy')
        indices = list(itertools.product(range(ARRAY_SIZE[0]), repeat=nDims))
        assert len(indices) == self.nElements
        for i in indices:
            self.arrProxy.ckInsert(i, [42, 'testing'])

        self.arrProxy.ckDoneInserting()
        self.arrProxy[firstIdx].SayHi(17)

    def done(self):
        print('Ring messaging done. Testing reduction.')
        self.arrProxy.TestReduction()

    def doneReduction(self, result):
        assert result == 1*self.nElements, 'Reduction for dynamic array insertion failed.'
        print('All done.')
        exit()


class Hello(Chare):

    def __init__(self, arg0, arg1):
        assert [arg0, arg1] == [42, 'testing'], 'Constructor args for dynamic array insertion failed.'
        print('Hello', self.thisIndex, 'created on PE', charm.myPe())

    def SayHi(self, hiNo):
        print('Hi[' + str(hiNo) + '] from element', self.thisIndex, 'on PE', charm.myPe())
        if self.thisIndex == lastIdx:
            mainProxy.done()
        else:
            nextIndex = list(self.thisIndex)
            for i in range(nDims-1,-1,-1):
                nextIndex[i] = (nextIndex[i] + 1) % ARRAY_SIZE[i]
                if nextIndex[i] != 0:
                    break
            return self.thisProxy[nextIndex].SayHi(hiNo+1)

    def TestReduction(self):
        self.contribute(1, Reducer.sum, mainProxy.doneReduction)


charm.start(Main)

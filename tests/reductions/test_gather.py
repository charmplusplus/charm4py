from charm4py import charm, Chare, Array, Group, Reducer


mainProxy = None


class Main(Chare):

    def __init__(self, args):
        self.recvdReductions = 0
        self.expectedReductions = 5

        nDims = 1
        ARRAY_SIZE = [10] * nDims
        self.nElements = 1
        for x in ARRAY_SIZE:
            self.nElements *= x
        print('Running gather example on', charm.numPes(), 'processors for', self.nElements, 'elements, array dims=', ARRAY_SIZE)
        arrProxy = Array(Test, ARRAY_SIZE)
        grpProxy = Group(TestGroup)
        charm.thisProxy.updateGlobals({'mainProxy': self.thisProxy}, '__main__', awaitable=True).get()
        arrProxy.doGather()
        grpProxy.doGather()
        red_future = charm.Future()
        arrProxy.doGather(red_future)
        self.done_gather_single(red_future.get())

    def done_gather_single(self, result):
        gather_arr_indices = list(range(self.nElements))
        gather_grp_indices = list(range(charm.numPes()))
        assert result == gather_arr_indices or result == gather_grp_indices, 'Gather single elements failed.'
        print('[Main] Gather collective for single elements done. Test passed')
        self.recvdReductions += 1
        if self.recvdReductions >= self.expectedReductions:
            exit()

    def done_gather_array(self, result):
        gather_arr_indices = [tuple([i]) for i in range(self.nElements)]
        gather_grp_indices = [[i, 42] for i in range(charm.numPes())]
        assert result == gather_arr_indices or result == gather_grp_indices, 'Gather arrays failed.'
        print('[Main] Gather collective for arrays done. Test passed')
        self.recvdReductions += 1
        if self.recvdReductions >= self.expectedReductions:
            exit()


class Test(Chare):

    def __init__(self):
        print('Test', self.thisIndex, 'created on PE', charm.myPe())

    def doGather(self, red_future=None):
        if red_future is None:
            # gather single elements
            self.contribute(self.thisIndex[0], Reducer.gather, mainProxy.done_gather_single)
            # gather arrays
            self.contribute(self.thisIndex, Reducer.gather, mainProxy.done_gather_array)
        else:
            self.contribute(self.thisIndex[0], Reducer.gather, red_future)


class TestGroup(Chare):

    def __init__(self):
        print('TestGroup', self.thisIndex, 'created on PE', charm.myPe())

    def doGather(self):
        # gather single elements
        self.contribute(self.thisIndex, Reducer.gather, mainProxy.done_gather_single)
        # gather arrays
        self.contribute([self.thisIndex, 42], Reducer.gather, mainProxy.done_gather_array)


charm.start(Main)

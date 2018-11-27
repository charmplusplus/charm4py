from charm4py import charm, Chare, Group, Array, ArrayMap
import itertools


def index_to_pe(index):
    return index.__hash__() % charm.numPes()


class MyMap(ArrayMap):

    def procNum(self, index):
        return index_to_pe(index)


class MyChare(Chare):

    def __init__(self, last):
        assert charm.myPe() == index_to_pe(self.thisIndex), "ArrayMap failed"
        if last: self.contribute(None, None, charm.thisProxy[0].exit)


def main(args):
    array_map = Group(MyMap)
    for nDims in range(1,7):
        if nDims <= 3:
            dim_size = 4
        else:
            dim_size = 2
        if nDims < 6:
            Array(MyChare, [dim_size]*nDims, args=[False], map=array_map)
        else:
            dyn_array = Array(MyChare, ndims=nDims, map=array_map)
            for idx in itertools.product(range(dim_size), repeat=nDims):
                dyn_array.ckInsert(idx, args=[True])
            dyn_array.ckDoneInserting()


charm.start(main)

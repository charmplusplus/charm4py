from charm4py import charm, Chare, Group, Array, Reducer
import numpy as np


class Test(Chare):
    def __init__(self, f):
        data = np.arange(10, dtype='float64')
        self.contribute(data, Reducer.sum, f)


def main(args):
    f1 = charm.createFuture()
    f2 = charm.createFuture()
    Group(Test, args=[f1])
    Array(Test, charm.numPes() * 4, args=[f2])
    np.testing.assert_allclose(f1.get(), np.arange(10, dtype='float64') * charm.numPes())
    np.testing.assert_allclose(f2.get(), np.arange(10, dtype='float64') * charm.numPes() * 4)
    exit()


charm.start(main)

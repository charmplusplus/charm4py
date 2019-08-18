from charm4py import charm, Chare, Channel, Future, coro
import numpy as np


NUM_ITER = 10


class Test(Chare):

    @coro
    def work(self, mainProxy, done_fut):
        ch = Channel(self, remote=mainProxy)
        for i in range(NUM_ITER):
            array1, array2, array3 = ch.recv()
            np.testing.assert_array_equal(array1, np.arange(100, dtype='int64') + i)
            np.testing.assert_array_equal(array2, np.arange(50, dtype='int64') + i)
            np.testing.assert_array_equal(array3, np.arange(70, dtype='int64') + i)
        done_fut()


class Main(Chare):

    def __init__(self, args):
        chare = Chare(Test, onPE=1)
        ch = Channel(self, remote=chare)
        done_fut = Future()
        chare.work(self.thisProxy, done_fut)
        for i in range(NUM_ITER):
            array1 = np.arange(100, dtype='int64') + i
            array2 = np.arange(50, dtype='int64') + i
            array3 = np.arange(70, dtype='int64') + i
            ch.send(array1, array2, array3)
        done_fut.get()
        exit()


charm.start(Main)

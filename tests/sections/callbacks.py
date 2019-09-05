from charm4py import charm, Chare, Group, Array, Reducer, Future
import numpy


def odd_idx_chares(obj):
    idx = obj.thisIndex
    if isinstance(idx, tuple):
        idx = obj.thisIndex[0]
    if idx % 2 == 1:
        return [0]
    else:
        return []


def assert_allclose(actual, desired, tol):
    assert len(actual) == len(desired)
    assert all([(abs(actual[i] - v) <= tol) for i, v in enumerate(desired)])


class Test(Chare):

    def __init__(self):
        if isinstance(self.thisIndex, tuple):
            self.idx = self.thisIndex[0]
        else:
            self.idx = self.thisIndex

    def setSecProxy(self, proxy):
        self.secProxy = proxy

    def setTest(self, future, expected):
        self.future = future
        self.expected = expected

    def recvResult(self, result=None):
        # print('GOT RESULT', result)
        assert self.idx % 2 == 1
        if isinstance(result, numpy.ndarray):
            assert_allclose(result, self.expected, 1e-01)
        else:
            assert result == self.expected
        self.contribute(1, Reducer.sum, self.future, self.secProxy)

    def work1(self, cb, secProxy=None):
        self.contribute(3, Reducer.sum, cb, secProxy)

    def work2(self, cb, secProxy=None):
        data = numpy.arange(100, dtype='float64')
        self.contribute(data, Reducer.sum, cb, secProxy)

    def work3(self, cb, secProxy=None):
        self.contribute(str(self.idx), Reducer.gather, cb, secProxy)

    def work4(self, cb, secProxy=None):
        self.contribute(None, None, cb, secProxy)

    def work5(self, cb, secProxy=None):
        if self.idx == 1:
            cb('test section callback')


def main(args):
    assert charm.numPes() % 2 == 0
    collections = []

    g = Group(Test)
    g_sec = charm.split(g, 1, odd_idx_chares)[0]
    g.setSecProxy(g_sec, awaitable=True).get()
    collections.append((g, g_sec, charm.numPes()))

    N = charm.numPes() * 10
    a = Array(Test, N)
    a_sec = charm.split(a, 1, odd_idx_chares)[0]
    a.setSecProxy(a_sec, awaitable=True).get()
    collections.append((a, a_sec, N))

    for collection, secProxy, numchares in collections:

        f = Future()
        expected = numchares * 3
        collection.setTest(f, expected, awaitable=True).get()
        collection.work1(secProxy.recvResult)
        assert f.get() == (numchares // 2)

        f = Future()
        expected = (numchares // 2) * 3
        secProxy.setTest(f, expected, awaitable=True).get()
        secProxy.work1(secProxy.recvResult, secProxy)
        assert f.get() == (numchares // 2)

        f = Future()
        expected = numpy.arange(100, dtype='float64')
        expected *= numchares
        collection.setTest(f, expected, awaitable=True).get()
        collection.work2(secProxy.recvResult)
        assert f.get() == (numchares // 2)

        f = Future()
        expected = numpy.arange(100, dtype='float64')
        expected *= (numchares // 2)
        secProxy.setTest(f, expected, awaitable=True).get()
        secProxy.work2(secProxy.recvResult, secProxy)
        assert f.get() == (numchares // 2)

        f = Future()
        expected = [str(i) for i in range(numchares)]
        collection.setTest(f, expected, awaitable=True).get()
        collection.work3(secProxy.recvResult)
        assert f.get() == (numchares // 2)

        f = Future()
        expected = [str(i) for i in range(numchares) if i % 2 == 1]
        secProxy.setTest(f, expected, awaitable=True).get()
        secProxy.work3(secProxy.recvResult, secProxy)
        assert f.get() == (numchares // 2)

        f = Future()
        expected = None
        collection.setTest(f, expected, awaitable=True).get()
        collection.work4(secProxy.recvResult)
        assert f.get() == (numchares // 2)

        f = Future()
        expected = None
        secProxy.setTest(f, expected, awaitable=True).get()
        secProxy.work4(secProxy.recvResult, secProxy)
        assert f.get() == (numchares // 2)

        f = Future()
        expected = 'test section callback'
        collection.setTest(f, expected, awaitable=True).get()
        collection.work5(secProxy.recvResult)
        assert f.get() == (numchares // 2)

        f = Future()
        expected = None
        collection.setTest(f, expected, awaitable=True).get()
        charm.startQD(secProxy.recvResult)
        assert f.get() == (numchares // 2)

    exit()


charm.start(main)

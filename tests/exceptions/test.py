from charm4py import charm, Chare, Group, Array, coro


class Test(Chare):

    def __init__(self):
        if isinstance(self.thisIndex, tuple):
            self.idx = self.thisIndex[0]
        else:
            self.idx = self.thisIndex

    def bad(self):
        if self.idx % 2 != 0:
            # this will raise NameError exception
            test[3] = 3
        else:
            return 'good'

    def allbad(self):
        # this will raise NameError exception
        test[3] = 3

    def good(self):
        return self.idx

    @coro
    def bad_th(self):
        return self.bad()

    @coro
    def allbad_th(self):
        return self.allbad()

    @coro
    def good_th(self):
        return self.good()


def main(args):
    assert charm.numPes() % 2 == 0

    NUM_ITER = 5
    npes = charm.numPes()
    g = Group(Test)
    a = Array(Test, npes * 8)

    for proxy, num_chares in ((g, npes), (a, npes * 8)):
        for i in range(2):
            if i == 0:
                methods = {'allbad': 'allbad', 'good': 'good', 'bad': 'bad'}
            else:
                methods = {'allbad': 'allbad_th', 'good': 'good_th', 'bad': 'bad_th'}

            # p2p
            if proxy == g:
                bad_idx = 1
            else:
                bad_idx = (num_chares // 2) + 1
            for _ in range(NUM_ITER):
                try:
                    getattr(proxy[bad_idx], methods['bad'])(ret=True).get()
                    assert False
                except NameError:
                    retval = getattr(proxy[bad_idx], methods['good'])(ret=True).get()
                    assert retval == bad_idx

            # bcast awaitable=True
            for _ in range(NUM_ITER):
                try:
                    getattr(proxy, methods['allbad'])(awaitable=True).get()
                    assert False
                except NameError:
                    try:
                        getattr(proxy, methods['bad'])(awaitable=True).get()
                        assert False
                    except NameError:
                        retval = getattr(proxy, methods['good'])(awaitable=True).get()
                        assert retval is None

            # bcast ret=True (returns list of results)
            for _ in range(NUM_ITER):
                retvals = getattr(proxy, methods['bad'])(ret=True).get()
                num_errors = 0
                for retval in retvals:
                    if isinstance(retval, NameError):
                        num_errors += 1
                    else:
                        assert retval == 'good'
                assert num_errors == (num_chares // 2)
    exit()


charm.start(main)

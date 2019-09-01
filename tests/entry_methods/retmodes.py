from charm4py import charm, Chare, Group, Array, Reducer, coro, Future
import random


class Test(Chare):

    def __init__(self, callback, callback_test_done):
        self.myval = random.randint(1, 100)
        self.rcvd_bcasts = 0
        self.callback_test_done = callback_test_done
        self.contribute(self.myval, Reducer.gather, callback)

    def getVal(self, bcast):
        if bcast:
            self.rcvd_bcasts += 1
            if self.rcvd_bcasts == 6:
                self.contribute(None, None, self.callback_test_done)
        return self.myval

    @coro
    def getVal_th(self, bcast):
        return self.getVal(bcast)


def main(args):
    f1 = Future()
    f2 = Future()
    done1 = Future()
    done2 = Future()
    a = Array(Test, charm.numPes() * 10, args=[f1, done1])
    g = Group(Test, args=[f2, done2])

    collections = []
    collections.append((a, f1.get(), charm.numPes() * 10))
    collections.append((g, f2.get(), charm.numPes()))
    for collection, vals, num_elems in collections:
        indexes = list(range(num_elems))
        random_idxs = random.sample(indexes, int(max(len(indexes) * 0.8, 1)))
        for random_idx in random_idxs:
            retval = collection[random_idx].getVal(False, awaitable=False, ret=False)
            assert retval is None

            retval = collection[random_idx].getVal(False, ret=True).get()
            assert retval == vals[random_idx]

            retval = collection[random_idx].getVal_th(False, awaitable=False, ret=False)
            assert retval is None

            retval = collection[random_idx].getVal_th(False, ret=True).get()
            assert retval == vals[random_idx]

        retval = collection.getVal(True, ret=False)
        assert retval is None

        retval = collection.getVal(True, awaitable=True).get()
        assert retval is None

        retval = collection.getVal(True, ret=True).get()
        assert retval == [vals[i] for i in range(num_elems)]

        retval = collection.getVal_th(True, awaitable=False)
        assert retval is None

        retval = collection.getVal_th(True, awaitable=True).get()
        assert retval is None

        retval = collection.getVal_th(True, ret=True).get()
        assert retval == [vals[i] for i in range(num_elems)]

    done1.get()
    done2.get()
    exit()


charm.start(main)

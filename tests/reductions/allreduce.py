from charm4py import charm, Chare, Group, Array, Reducer, threaded


class Test(Chare):

    @threaded
    def run(self, numChares, done_future):
        if isinstance(self.thisIndex, tuple):
            index = self.thisIndex[0]
        else:
            index = self.thisIndex
        i = 137
        for _ in range(1000):
            result = charm.allReduce(index + i, Reducer.sum, self)
            expected = float(numChares) * (numChares - 1) / 2 + (i * numChares)
            assert result == expected
            i += 1

        self.contribute(None, None, done_future)


def main(args):
    g1 = Group(Test)
    g2 = Group(Test)

    numArrayChares = charm.numPes() * 8
    a1 = Array(Test, numArrayChares)
    a2 = Array(Test, numArrayChares)

    charm.awaitCreation(g1, g2, a1, a2)

    wait_alldone = [charm.createFuture() for _ in range(4)]
    i = 0
    for collection in (g1, g2):
        collection.run(charm.numPes(), wait_alldone[i])
        i += 1

    for collection in (a1, a2):
        collection.run(numArrayChares, wait_alldone[i])
        i += 1

    for done in wait_alldone:
        done.get()
    print('DONE')
    exit()


charm.start(main)

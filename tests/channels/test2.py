from charm4py import charm, Chare, Group, Array, Channel, coro, Future, Reducer
from collections import defaultdict
import random


NUM_LEVELS = 3
LEVELS_START = (37, 533, 17000)
LEVELS_NUM_ITER = (30, 50, 10)
NUM_CHANNELS = 1000


class Test(Chare):

    def setup(self):
        self.channels = {}
        self.nbs = {}
        me = self.thisProxy[self.thisIndex]
        for level in range(NUM_LEVELS):
            self.channels[level] = []
            self.nbs[level] = set()
            for nb in gchannels[level][me]:
                self.channels[level].append(Channel(self, remote=nb))
                self.nbs[level].add(nb)

    @coro
    def work(self, level, done_fut):
        msgs = 0
        start = LEVELS_START[level]
        me = self.thisProxy[self.thisIndex]
        channels = list(self.channels[level])
        for i in range(LEVELS_NUM_ITER[level]):
            random.shuffle(channels)
            for ch in channels:
                ch.send(me, start + i)
            for ch in charm.iwait(channels):
                remote, data = ch.recv()
                assert data == start + i
                assert ch.remote == remote
                assert remote in self.nbs[level]
                msgs += 1
        self.reduce(done_fut, msgs, Reducer.sum)


def main(args):
    g1 = Group(Test)
    g2 = Group(Test)
    g3 = Group(Test)
    g4 = Group(Test)

    P = charm.numPes()
    a1 = Array(Test, P * 8)
    a2 = Array(Test, P * 10)
    a3 = Array(Test, P * 4)
    a4 = Array(Test, P * 1)

    charm.awaitCreation(g1, g2, g3, g4, a1, a2, a3, a4)

    chares = []  # proxies to all individual chares created
    for collection in (g1, g2, g3, g4):
        for idx in range(P):
            chares.append(collection[idx])

    for collection, numelems in ((a1, P*8), (a2, P*10), (a3, P*4), (a4, P)):
        for idx in range(numelems):
            chares.append(collection[idx])

    print('There are', len(chares), 'chares')

    # establish random channels between chares
    global gchannels
    gchannels = {}
    num_self_channels = 0
    for level in range(NUM_LEVELS):
        gchannels[level] = defaultdict(list)
        for _ in range(NUM_CHANNELS):
            a = random.choice(chares)
            b = random.choice(chares)
            if a == b:
                num_self_channels += 1
            gchannels[level][a].append(b)
            gchannels[level][b].append(a)
    charm.thisProxy.updateGlobals({'gchannels': gchannels}, awaitable=True).get()

    done_fut = Future(8 * NUM_LEVELS)  # wait for 8 collections to finish 3 levels
    for collection in (g1, g2, g3, g4, a1, a2, a3, a4):
        collection.setup(awaitable=True).get()
    print(NUM_CHANNELS * NUM_LEVELS, 'channels set up,', num_self_channels, 'self channels')
    for collection in (g1, g2, g3, g4, a1, a2, a3, a4):
        for lvl in range(NUM_LEVELS):
            collection.work(lvl, done_fut)

    msgs = sum(done_fut.get())
    assert msgs == sum(LEVELS_NUM_ITER[:NUM_LEVELS]) * NUM_CHANNELS * 2
    print('total msgs received by chares=', msgs)
    exit()


charm.start(main)

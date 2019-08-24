from charm4py import charm, Chare, Group, Channel
import time


class Test(Chare):

    def work(self, mainProxy):
        time.sleep(self.thisIndex * 0.5)
        ch = Channel(self, remote=mainProxy)
        ch.send(self.thisIndex)


class Main(Chare):

    def __init__(self, args):
        N = min(4, charm.numPes())
        g = Group(Test)
        channels = [Channel(self, g[i]) for i in range(N)]
        for i in range(N):
            g[i].work(self.thisProxy)

        channels.reverse()

        t0 = time.time()
        idx = 0
        for ch in charm.iwait(channels):
            assert ch.recv() == idx
            idx += 1
            print(time.time() - t0)
        assert idx == N
        exit()


charm.start(Main)

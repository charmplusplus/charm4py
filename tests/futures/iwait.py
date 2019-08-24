from charm4py import charm, Chare, Group, Future
import time


class Test(Chare):

    def work(self, fut):
        time.sleep(self.thisIndex * 0.5)
        fut(self.thisIndex)


def main(args):
    N = min(4, charm.numPes())
    g = Group(Test)
    futures = [Future() for _ in range(N)]
    for i in range(N):
        g[i].work(futures[i])

    futures.reverse()

    t0 = time.time()
    idx = 0
    for f in charm.iwait(futures):
        assert f.get() == idx
        idx += 1
        print(time.time() - t0)
    assert idx == N
    exit()


charm.start(main)

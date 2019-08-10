from charm4py import charm, Chare, Group, Array, Future
from charm4py import threads
from time import time

CHARES_PER_PE = 8
LOAD = 0.03  # in secs
WORK_TIME = CHARES_PER_PE * LOAD


class Worker(Chare):

    def __init__(self, numChares):
        idx = self.thisIndex[0]
        self.nbs = []
        self.nbs.append(self.thisProxy[(idx + 1) % numChares])
        self.nbs.append(self.thisProxy[(idx + 2) % numChares])
        self.nbs.append(self.thisProxy[(idx - 1) % numChares])
        self.nbs.append(self.thisProxy[(idx - 2) % numChares])

    def start(self):
        self.msgsRcvd = 0
        t0 = time()
        x = 0
        while time() - t0 < LOAD:
            x += 1
        for nb in self.nbs:
            nb.recvMsg()

    def recvMsg(self):
        self.msgsRcvd += 1

    def check(self, cb):
        assert self.msgsRcvd == len(self.nbs)
        self.contribute(None, None, cb)
        self.msgsRcvd = 0


class QDReceiver(Chare):

    def __init__(self, mainProxy):
        self.mainProxy = mainProxy
        self.firstTime = True

    def recvQD(self):
        if self.firstTime:
            # QD callback sent to every element in this collection
            self.firstTime = False
            self.contribute(None, None, self.mainProxy.recvQD)
        else:
            # QD callback was only to me
            assert self.thisIndex == 1 or self.thisIndex == (1,)
            self.mainProxy.recvQD()


class Main(Chare):

    def __init__(self, args):
        assert charm.numPes() > 1
        numChares = charm.numPes() * CHARES_PER_PE
        self.workers = Array(Worker, numChares, args=[numChares])
        print('WORK_TIME=', WORK_TIME)
        qdGroupReceivers = Group(QDReceiver, args=[self.thisProxy])
        qdArrayReceivers = Array(QDReceiver, charm.numPes(), args=[self.thisProxy])
        charm.awaitCreation(self.workers, qdGroupReceivers, qdArrayReceivers)

        self.testQD(callback=self.thisProxy.recvQD)
        self.testQD(callback=qdGroupReceivers.recvQD)
        self.testQD(callback=qdArrayReceivers.recvQD)
        self.testQD(callback=qdGroupReceivers[1].recvQD)
        self.testQD(callback=qdArrayReceivers[1].recvQD)
        self.testQD(callback=Future())
        self.testQD(callback=None)

        exit()

    def testQD(self, callback):
        self.qdReached = False
        check_fut = Future()
        t0 = time()
        self.workers.start()
        if callback is not None:
            charm.startQD(callback)
            if isinstance(callback, threads.Future):
                callback.get()
                print('QD reached')
            else:
                self.wait('self.qdReached')
        else:
            charm.waitQD()
        assert time() - t0 > WORK_TIME
        self.workers.check(check_fut)
        check_fut.get()

    def recvQD(self):
        print('QD reached')
        self.qdReached = True


charm.start(Main)

from charm4py import charm, Chare, Array, threaded
from time import time
import numpy as np

PAYLOAD = 100  # number of bytes
NITER = 10000


class Ping(Chare):

    def __init__(self):
        self.myIndex = self.thisIndex[0]
        if self.myIndex == 0:
            self.neighbor = self.thisProxy[1]
        else:
            self.neighbor = self.thisProxy[0]

    def start(self, done_future, threaded=False):
        self.done_future = done_future
        self.iter = 0
        #data = np.zeros(PAYLOAD, dtype='int8')
        data = 3
        self.startTime = time()
        if threaded:
            self.neighbor.recv_th(data)
        else:
            self.neighbor.recv(data)

    def recv(self, data):
        if self.myIndex == 0:
            self.iter += 1
            if self.iter == NITER:
                totalTime = time() - self.startTime
                self.done_future.send(totalTime)
                return
        self.neighbor.recv(data)

    @threaded
    def recv_th(self, data):
        if self.myIndex == 0:
            self.iter += 1
            if self.iter == NITER:
                totalTime = time() - self.startTime
                self.done_future.send(totalTime)
                return
        self.neighbor.recv_th(data)


def main(args):
    threaded = False
    if len(args) > 1 and args[1] == '-t':
        threaded = True
    pings = Array(Ping, 2)
    charm.awaitCreation(pings)
    for _ in range(2):
        done_future = charm.createFuture()
        pings[0].start(done_future, threaded)
        totalTime = done_future.get()
        print("ping pong time per iter (us)=", totalTime / NITER * 1000000)
    exit()


charm.start(main)

from charm4py import charm, Chare, Array, coro, Future
from time import time
import numpy as np
import sys

ITER=2000
WARM_UP=200

class Ping(Chare):

    def __init__(self):
        print("My PE: ", charm.myPe())
        self.myIndex = self.thisIndex[0]
        if self.myIndex == 0:
            self.neighbor = self.thisProxy[1]
            print("[0] neighbor = ", self.neighbor)
        else:
            self.neighbor = self.thisProxy[0]
            print("[1] neighbor = ", self.neighbor)

    def start(self, done_future, threaded, length):
        self.done_future = done_future
        self.iter = 0
        self.length = length
        self.data = np.empty(length, dtype='i') #int
        data = np.arange(length, dtype='i')

        self.startTime = time()
        self.neighbor.recv(data)

    def recv(self, data):
        if self.myIndex == 0:
            self.iter += 1
            if self.iter == WARM_UP:
                self.startTime = time()
            if self.iter == ITER + WARM_UP:
                totalTime = (time() - self.startTime) / ITER
                totalTime /= 2
                self.done_future.send(totalTime)
                return
        self.neighbor.recv(data)

def main(args):
    threaded = False
    threaded_time = []
    plain_time = []
    pings = Array(Ping, 2)
    charm.awaitCreation(pings)
    for l in range(15):
        length = 2**l
        print("Array length = {}".format(length))

        for _ in range(1):
            #threaded = not threaded
            done_future = Future()
            pings[0].start(done_future, threaded, length)
            totalTime = done_future.get()
            print("RES:: [plain] {} ".format(length), totalTime * 1e6)
    exit()

charm.start(main)

from charm4py import charm, Chare, Array, coro, Future
from time import time
import numpy as np
from numba import cuda

PAYLOAD = 100  # number of bytes
NITER = 10000


class Ping(Chare):

    def __init__(self, gpu, num_iters):
        self.gpu = gpu
        self.myIndex = self.thisIndex[0]
        if self.myIndex == 0:
            self.neighbor = self.thisProxy[1]
        else:
            self.neighbor = self.thisProxy[0]

    def start(self, done_future, payload_size):
        self.done_future = done_future
        self.iter = 0
        data = np.zeros(payload_size, dtype='int8')
        if self.gpu:
            data = cuda.to_device(data)
            self.startTime = time()

        else:
            self.neighbor.recv(data)

    def recv(self, data):
        data = cuda.to_device(data)
        if self.myIndex == 0:
            self.iter += 1
            if self.iter == NITER:
                totalTime = time() - self.startTime
                self.done_future.send(totalTime)
                return
        data = data.copy_to_host()
        self.neighbor.recv(data)

    @coro
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
    gpu = False
    min_msg_size, max_mig_size, low_iter, high_iter, printFormat, gpu = 0
    if len(args) < 7:
        print("Doesn't have the required input params. Usage:"
              "<max-msg-size> <low-iter> <high-iter> <print-format"
              "(0 for csv, 1 for "
              "regular)> <GPU (0 for CPU, 1 for GPU)>\n"
              )
        charm.exit(-1)

    min_msg_size = int(args[1])
    max_msg_size = int(args[2])
    low_iter = int(args[3])
    high_iter = int(args[4])
    print_format = int(args[5])
    gpu = int(args[6])

    pings = Array(Ping, 2, gpu)
    charm.awaitCreation(pings)
    for _ in range(2):
        done_future = Future()
        pings[0].start(done_future, threaded, gpu)
        totalTime = done_future.get()
        print("ping pong time per iter (us)=", totalTime / NITER * 1000000)
    exit()


charm.start(main)

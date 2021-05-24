from charm4py import charm, Chare, Array, coro, Future, Channel, Group, ArrayMap
import time
import numpy as np
import array

LOW_ITER_THRESHOLD = 8192
WARMUP_ITERS = 10


class Block(Chare):
    def __init__(self):
        self.num_chares = charm.numPes()
        self.am_low_chare = self.thisIndex[0] == 0

        if self.am_low_chare:
            print("Msg Size, Iterations, Bandwidth (MB/s)")

    @coro
    def do_iteration(self, message_size, windows, num_iters, done_future):
        local_data = np.ones(message_size, dtype='int8')
        remote_data = np.ones(message_size, dtype='int8')

        partner_idx = int(not self.thisIndex[0])
        partner = self.thisProxy[partner_idx]
        partner_channel = Channel(self, partner)
        partner_ack_channel = Channel(self, partner)

        tstart = 0

        for idx in range(num_iters + WARMUP_ITERS):
            if idx == WARMUP_ITERS:
                tstart = time.time()
            if self.am_low_chare:
                for _ in range(windows):
                    partner_channel.send(local_data)
                partner_ack_channel.recv()
            else:
                for _ in range(windows):
                    partner_channel.recv()
                partner_ack_channel.send(1)

        tend = time.time()
        elapsed_time = tend - tstart
        if self.am_low_chare:
            self.display_iteration_data(elapsed_time, num_iters, windows, message_size)

        self.reduce(done_future)

    def display_iteration_data(self, elapsed_time, num_iters, windows, message_size):
        data_sent = message_size / 1e6 * num_iters * windows;
        print(f'{message_size},{num_iters},{data_sent/elapsed_time}')



class ArrMap(ArrayMap):
    def procNum(self, index):
        return index[0] % 2


def main(args):
    if len(args) < 6:
        print("Doesn't have the required input params. Usage:"
              "<min-msg-size> <max-msg-size> <window-size> "
              "<low-iter> <high-iter>\n"
              )
        charm.exit(-1)

    min_msg_size = int(args[1])
    max_msg_size = int(args[2])
    window_size = int(args[3])
    low_iter = int(args[4])
    high_iter = int(args[5])

    peMap = Group(ArrMap)
    blocks = Array(Block, 2, map = peMap)
    charm.awaitCreation(blocks)
    msg_size = min_msg_size

    while msg_size <= max_msg_size:
        if msg_size <= LOW_ITER_THRESHOLD:
            iter = low_iter
        else:
            iter = high_iter
        done_future = Future()
        blocks.do_iteration(msg_size, window_size, iter, done_future)
        done_future.get()
        msg_size *= 2

    charm.exit()


charm.start(main)

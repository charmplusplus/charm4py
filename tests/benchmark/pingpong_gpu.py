from charm4py import charm, Chare, Array, coro, Future, Channel, Group, ArrayMap
import time
import numpy as np
from numba import cuda

USE_PINNED = True
WARMUP_ITERS = 10

class Ping(Chare):
    def __init__(self, use_gpudirect, print_format):
        self.gpu_direct = use_gpudirect
        self.num_chares = charm.numPes()
        self.print_format = print_format
        self.am_low_chare = self.thisIndex[0] == 0

        if self.am_low_chare:
            if print_format == 0:
                print("Msg Size, Iterations, One-way Time (us), "
                      "Bandwidth (bytes/us)"
                      )
            else:
                print(f'{"Msg Size": <30} {"Iterations": <25} '
                      f'{"One-way Time (us)": <20} {"Bandwidth (bytes/us)": <20}'
                      )

    @coro
    def do_iteration(self, message_size, num_iters, done_future):
        if USE_PINNED:
            h_data_send = cuda.pinned_array(message_size, dtype='int8')
            h_data_recv = cuda.pinned_array(message_size, dtype='int8')
        else:
            if self.am_low_chare:
                h_data_send = np.ones(message_size, dtype='int8')
                h_data_recv = np.ones(message_size, dtype='int8')
            else:
                h_data_send = np.zeros(message_size, dtype='int8')
                h_data_recv = np.zeros(message_size, dtype='int8')

        d_data_send = cuda.device_array(message_size, dtype='int8')
        d_data_recv = cuda.device_array(message_size, dtype='int8')
        d_data_send.copy_to_device(h_data_send)
        d_data_recv.copy_to_device(h_data_recv)
        partner_idx = int(not self.thisIndex[0])
        partner = self.thisProxy[partner_idx]
        partner_channel = Channel(self, partner)

        for _ in range(WARMUP_ITERS):
            if self.am_low_chare:
                if not self.gpu_direct:
                    d_data_send.copy_to_host(h_data_send)
                    partner_channel.send(h_data_send)
                    d_data_recv.copy_to_device(partner_channel.recv())
                else:
                    partner_channel.send(d_data_send)
                    partner_channel.recv(d_data_recv)

            else:
                if not self.gpu_direct:
                    d_data_recv.copy_to_device(partner_channel.recv())
                    d_data_send.copy_to_host(h_data_send)
                    partner_channel.send(h_data_send)
                else:
                    partner_channel.recv(d_data_recv)
                    partner_channel.send(d_data_send)


        tstart = time.time()

        for _ in range(num_iters):
            if self.am_low_chare:
                if not self.gpu_direct:
                    d_data_send.copy_to_host(h_data_send)
                    partner_channel.send(h_data_send)
                    d_data_recv.copy_to_device(partner_channel.recv())
                else:
                    partner_channel.send(d_data_send)
                    partner_channel.recv(d_data_recv)

            else:
                if not self.gpu_direct:
                    d_data_recv.copy_to_device(partner_channel.recv())
                    d_data_send.copy_to_host(h_data_send)
                    partner_channel.send(h_data_send)
                else:
                    partner_channel.recv(d_data_recv)
                    partner_channel.send(d_data_send)

        tend = time.time()

        elapsed_time = tend - tstart

        if self.am_low_chare:
            self.display_iteration_data(elapsed_time, num_iters, message_size)

        self.reduce(done_future)

    def display_iteration_data(self, elapsed_time, num_iters, message_size):
        elapsed_time /= 2  # 1-way performance, not RTT
        elapsed_time /= num_iters  # Time for each message
        bandwidth = message_size / elapsed_time
        if self.print_format == 0:
            print(f'{message_size},{num_iters},{elapsed_time * 1e6},'
                  f'{bandwidth / 1e6}'
                  )
        else:
            print(f'{message_size: <30} {num_iters: <25} '
                  f'{elapsed_time * 1e6: <20} {bandwidth / 1e6: <20}'
                  )


class ArrMap(ArrayMap):
    def procNum(self, index):
        return index[0] % 2


def main(args):
    if len(args) < 7:
        print("Doesn't have the required input params. Usage:"
              "<min-msg-size> <max-msg-size> <low-iter> "
              "<high-iter> <print-format"
              "(0 for csv, 1 for "
              "regular)> <GPU (0 for host staging, 1 for GPU Direct)>\n"
              )
        charm.exit(-1)

    min_msg_size = int(args[1])
    max_msg_size = int(args[2])
    low_iter = int(args[3])
    high_iter = int(args[4])
    print_format = int(args[5])
    use_gpudirect = int(args[6])

    peMap = Group(ArrMap)
    pings = Array(Ping, 2, args=[use_gpudirect, print_format], map = peMap)
    charm.awaitCreation(pings)
    msg_size = min_msg_size

    # do a warmup iteration (should this be done for each size?)
    done_future = Future()
    pings.do_iteration(msg_size, high_iter, done_future)
    done_future.get()


    while msg_size <= max_msg_size:
        if msg_size <= 1048576:
            iter = low_iter
        else:
            iter = high_iter
        done_future = Future()
        pings.do_iteration(msg_size, iter, done_future)
        done_future.get()
        msg_size *= 2

    charm.exit()


charm.start(main)

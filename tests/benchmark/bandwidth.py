from charm4py import charm, Chare, Array, coro, Future, Channel, Group, ArrayMap
import time
import numpy as np
from numba import cuda
import array

USE_PINNED = True
# provide the address/size data for GPU-direct addresses. Saves ~11us per iteration
USE_ADDRESS_OPTIMIZATION = True
LOW_ITER_THRESHOLD = 8192
WARMUP_ITERS = 10


class Block(Chare):
    def __init__(self, use_gpudirect):
        self.gpu_direct = use_gpudirect
        self.num_chares = charm.numPes()
        self.am_low_chare = self.thisIndex[0] == 0

        if self.am_low_chare:
            print("Msg Size, Iterations, Bandwidth (MB/s)")

    @coro
    def do_iteration(self, message_size, windows, num_iters, done_future):
        if USE_PINNED:
            h_local_data = cuda.pinned_array(message_size, dtype='int8')
            h_remote_data = cuda.pinned_array(message_size, dtype='int8')
        else:
            if self.am_low_chare:
                h_local_data = np.ones(message_size, dtype='int8')
                h_remote_data = np.ones(message_size, dtype='int8')
            else:
                h_local_data = np.zeros(message_size, dtype='int8')
                h_remote_data = np.zeros(message_size, dtype='int8')


        d_local_data = cuda.device_array(message_size, dtype='int8')
        d_remote_data = cuda.device_array(message_size, dtype='int8')

        my_stream = cuda.stream()
        stream_address = my_stream.handle.value

        d_local_data_addr = d_local_data.__cuda_array_interface__['data'][0]
        h_local_data_addr = h_local_data.__array_interface__['data'][0]

        d_remote_data_addr = d_remote_data.__cuda_array_interface__['data'][0]
        h_remote_data_addr = h_remote_data.__array_interface__['data'][0]

        if self.gpu_direct:
            d_local_data_addr = array.array('L', [0])
            d_local_data_size = array.array('L', [0])

            d_local_data_addr[0] = d_local_data.__cuda_array_interface__['data'][0]
            d_local_data_size[0] = d_local_data.nbytes


        partner_idx = int(not self.thisIndex[0])
        partner = self.thisProxy[partner_idx]
        partner_channel = Channel(self, partner)
        partner_ack_channel = Channel(self, partner)

        tstart = 0

        for idx in range(num_iters + WARMUP_ITERS):
            if idx == WARMUP_ITERS:
                tstart = time.time()
            if self.am_low_chare:
                if not self.gpu_direct:
                    for _ in range(windows):
                        charm.lib.CudaDtoH(h_local_data_addr, d_local_data_addr, message_size, stream_address)
                    charm.lib.CudaStreamSynchronize(stream_address)
                        # d_local_data.copy_to_host(h_local_data)
                    for _ in range(windows):
                        partner_channel.send(h_local_data)
                else:
                    for _ in range(windows):
                        partner_channel.send(gpu_src_ptrs = d_local_data_addr,
                                             gpu_src_sizes = d_local_data_size
                                             )

                partner_ack_channel.recv()
            else:
                if not self.gpu_direct:
                    for _ in range(windows):
                        received = partner_channel.recv()
                        charm.lib.CudaHtoD(d_remote_data_addr, received.__array_interface__['data'][0],
                                           message_size, stream_address
                                           )
                    charm.lib.CudaStreamSynchronize(stream_address)
                        # d_local_data.copy_to_device(received)
                else:
                    for _ in range(windows):
                        partner_channel.recv(post_buf_addresses = d_local_data_addr,
                                             post_buf_sizes = d_local_data_size)
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
    if len(args) < 7:
        print("Doesn't have the required input params. Usage:"
              "<min-msg-size> <max-msg-size> <window-size> "
              "<low-iter> <high-iter>"
              "<GPU (0 for host staging, 1 for GPU Direct)>\n"
              )
        charm.exit(-1)

    min_msg_size = int(args[1])
    max_msg_size = int(args[2])
    window_size = int(args[3])
    low_iter = int(args[4])
    high_iter = int(args[5])
    use_gpudirect = int(args[6])

    peMap = Group(ArrMap)
    blocks = Array(Block, 2, args=[use_gpudirect], map = peMap)
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

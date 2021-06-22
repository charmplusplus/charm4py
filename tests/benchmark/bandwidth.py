from charm4py import charm, Chare, Array, coro, Future, Channel, Group, ArrayMap
import time
import numpy as np
import array
from pypapi import papi_high
from pypapi import events as papi_events
import pandas as pd


LOW_ITER_THRESHOLD = 8192
WARMUP_ITERS = 10


class Block(Chare):
    def __init__(self, min_data, max_data):
        self.num_chares = charm.numPes()
        self.am_low_chare = self.thisIndex[0] == 0
        self.datarange = (min_data, max_data)
        self.output_df = None

        if self.am_low_chare:
            print("Chare,Msg Size, Iterations, Bandwidth (MB/s), L2 Miss Rate, L3 Miss Rate, L2 Misses, L2 Accesses, L3 Misses, L3 Accesses, Min, Max, Mean, Stdev")
        counters = [papi_events.PAPI_L2_TCM,
                    papi_events.PAPI_L3_TCM,
                    papi_events.PAPI_L2_TCA,
                    papi_events.PAPI_L3_TCA
                    ]
        self.ncounters = len(counters)
        papi_high.start_counters(counters)

    @coro
    def do_iteration(self, message_size, windows, num_iters, done_future, iter_datafile_base):
        local_data = np.ones(message_size, dtype='int8')
        remote_data = np.ones(message_size, dtype='int8')
        t_data = np.zeros(num_iters+WARMUP_ITERS, dtype='float64')
        papi_data = np.zeros((num_iters+WARMUP_ITERS, self.ncounters), dtype='int64')

        partner_idx = int(not self.thisIndex[0])
        partner = self.thisProxy[partner_idx]
        partner_channel = Channel(self, partner)
        partner_ack_channel = Channel(self, partner)

        tstart = 0

        for idx in range(num_iters + WARMUP_ITERS):
            if idx == WARMUP_ITERS:
                papi_high.read_counters()
                tstart = time.time()
            tst = time.time()
            if self.am_low_chare:
                for _ in range(windows):
                    partner_channel.send(local_data)
                partner_ack_channel.recv()
            else:
                for _ in range(windows):
                    # The lifetime of this object has big impact on performance
                    d = partner_channel.recv()
                partner_ack_channel.send(1)
            tend=time.time()
            t_data[idx] = tend-tst
            papi_data[idx] += papi_high.read_counters()

        # if self.am_low_chare:

        tend = time.time()
        elapsed_time = tend - tstart
        # if self.am_low_chare:
        self.display_iteration_data(elapsed_time, num_iters, windows, message_size, papi_data, t_data)
        if iter_datafile_base:
            iter_filename = iter_datafile_base + str(self.thisIndex[0]) + '.csv'
            iter_data = self.write_iteration_data(num_iters, windows, message_size, papi_data, t_data)
            if self.output_df is None:
                self.output_df = iter_data
            else:
                self.output_df = pd.concat([self.output_df, iter_data])
            if message_size == self.datarange[1]:
                self.output_df.to_csv(iter_filename, index=False)

        self.reduce(done_future)

    def write_iteration_data(self, num_iters, windows, message_size, papi_data, timing_data):
        assert len(papi_data) == len(timing_data) == num_iters + WARMUP_ITERS
        header = ("Chare,Msg Size, Iteration, Bandwidth (MB/s), "
                  "L2 Miss Rate, L3 Miss Rate, L2 Misses, L2 Accesses, "
                  "L3 Misses, L3 Accesses"
                  )
        output = pd.DataFrame(columns=header.split(','))

        timing_nowarmup = timing_data[WARMUP_ITERS::]
        papi_nowarmup = papi_data[WARMUP_ITERS::]
        per_iter_data_sent = message_size / 1e6 * windows
        for papi, elapsed_s, iteration in zip(papi_nowarmup,
                                              timing_nowarmup,
                                              range(num_iters)
                                              ):
            l2_tcm, l3_tcm, l2_tca, l3_tca = papi
            bandwidth = (per_iter_data_sent) / elapsed_s
            iter_num = iteration + 1

            iter_data = (self.thisIndex[0], message_size, iter_num,
                         bandwidth, l2_tcm/l2_tca, l3_tcm/l3_tca,
                         l2_tcm, l2_tca, l3_tcm, l3_tca
                         )
            output.loc[iteration] = iter_data
        return output


    def display_iteration_data(self, elapsed_time, num_iters, windows, message_size, papi_data,timing):
        from scipy import stats
        import math
        l2_tcm, l3_tcm, l2_tca, l3_tca = sum(papi_data)
        data_sent = message_size / 1e6 * num_iters * windows
        timing_nowarmup = timing[WARMUP_ITERS::]
        bandwidth_nowarmup = (message_size/1e6*windows) / timing_nowarmup
        descriptive = stats.describe(bandwidth_nowarmup)
        minmax, mean, variance = descriptive.minmax, descriptive.mean, descriptive.variance
        min, max = minmax
        print(f'{self.thisIndex[0]},{message_size},{num_iters},{elapsed_time},{data_sent/elapsed_time},{l2_tcm/l2_tca},{l3_tcm/l3_tca},{l2_tcm},{l2_tca},{l3_tcm},{l3_tca},{min},{max},{mean},{math.sqrt(variance)}')



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
    if len(args) == 7:
        iter_datafile_base = args[6]
    else:
        iter_datafile_base = None

    peMap = Group(ArrMap)
    blocks = Array(Block, 2, args=[min_msg_size, max_msg_size], map = peMap)
    charm.awaitCreation(blocks)
    msg_size = min_msg_size

    while msg_size <= max_msg_size:
        if msg_size <= LOW_ITER_THRESHOLD:
            iter = low_iter
        else:
            iter = high_iter
        done_future = Future()
        blocks.do_iteration(msg_size, window_size, iter, done_future, iter_datafile_base)
        done_future.get()
        msg_size *= 2

    charm.exit()


charm.start(main)

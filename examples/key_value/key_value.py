from charm4py import charm, Chare, Array, Future, Reducer
import math
import random
import time


class Table(Chare):
    def __init__(self, main_proxy, num_chares, k):
        self.main_proxy = main_proxy
        self.num_chares = num_chares
        self.k = k
        self.table = {}

        for i in range(k):
            key = self.thisIndex[0] * (k // num_chares) + i
            self.table[key] = math.pow(key, 1.0 / 3.0)

    def get(self, key, client_proxy):
        value = self.table[key]
        client_proxy.receive(key, value)


class Client(Chare):
    def __init__(self, table, max_requests_per_chare, k):
        self.num_requests = random.randint(1, max_requests_per_chare)
        self.satisfied_requests = 0
        self.requested_values = list()
        self.k = k
        self.table_proxy = table

    def run(self, done_future):
        self.done_fut = done_future
        for i in range(self.num_requests):
            key = random.randint(0, self.k - 1)
            table_idx = key // self.k
            self.table_proxy[table_idx].get(key, self.thisProxy[self.thisIndex])

    def receive(self, key, value):
        self.satisfied_requests += 1
        if self.satisfied_requests == self.num_requests:
            self.reduce(self.done_fut, self.num_requests, Reducer.sum)


def main(args):
    num_table_chares = 10
    num_client_chares = 25
    total_table_values = 10000
    max_requests_per_chare = 1000
    table = Array(Table, num_table_chares, args=[charm.thisProxy, num_table_chares, total_table_values])
    clients = Array(Client, num_client_chares, args=[table, max_requests_per_chare, total_table_values])
    done_fut = Future()

    tst = time.time()
    clients.run(done_fut)
    total_requests = done_fut.get()
    tend = time.time()

    print(f'Finished {total_requests} requests in {tend - tst:1.2f} seconds, '
          f'{total_requests / (tend - tst):1.2f} requests per second')
    charm.exit()


charm.start(main)

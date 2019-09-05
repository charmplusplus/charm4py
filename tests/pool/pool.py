from charm4py import charm, coro, Future
from time import time


NUM_TRIALS = 10


def square(x):
    return x**2


@coro
def square_coro(x):
    f = Future()
    return x**2


def add_val(x):
    return x + 5


@coro
def add_val_coro(x):
    f = Future()
    return x + 5


def main(args):
    NUM_TASKS = (charm.numPes() - 1) * 100

    # test charm.pool.map()
    tasks = list(range(NUM_TASKS))
    for chunksize in (1, 8):
        for func in (square, square_coro):
            t0 = time()
            for _ in range(NUM_TRIALS):
                result = charm.pool.map(func, tasks, chunksize=chunksize)
                assert result == [func(x) for x in tasks]
            print('Elapsed=', time() - t0)

    # test charm.pool.submit()
    funcs = [square, square_coro, add_val, add_val_coro]
    tasks = []
    for i in range(NUM_TASKS):
        tasks.append((funcs[i % len(funcs)], i))
    for chunksize in (1, 8):
        t0 = time()
        for _ in range(NUM_TRIALS):
            result = charm.pool.submit(tasks, chunksize=chunksize)
            assert result == [f(x) for f, x in tasks]
        print('Elapsed=', time() - t0)

    exit()


charm.start(main)

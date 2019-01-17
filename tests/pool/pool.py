from charm4py import charm
from time import time


NUM_TRIALS = 10


def square(x):
    return x**2


def add_val(x):
    return x + 5


def main(args):
    NUM_TASKS = (charm.numPes() - 1) * 100

    # test charm.pool.map()
    tasks = list(range(NUM_TASKS))
    for chunksize in (1, 8):
        for nested in (False, True):
            t0 = time()
            for _ in range(NUM_TRIALS):
                result = charm.pool.map(square, tasks, chunksize=chunksize, allow_nested=nested)
                assert result == [square(x) for x in tasks]
            print('Elapsed=', time() - t0)

    # test charm.pool.submit()
    tasks = []
    for i in range(NUM_TASKS):
        if i % 2 == 0:
            tasks.append((square, i))
        else:
            tasks.append((add_val, i))
    for chunksize in (1, 8):
        for nested in (False, True):
            t0 = time()
            for _ in range(NUM_TRIALS):
                result = charm.pool.submit(tasks, chunksize=chunksize, allow_nested=nested)
                assert result == [f(x) for f,x in tasks]
            print('Elapsed=', time() - t0)

    exit()


charm.start(main)

from charm4py import charm


def square(x):
    return x ** 2


def add_val(x):
    return x + 5


def main(args):

    assert charm.numPes() >= 4, "Run this test with at least 4 PEs"

    tasks1 = list(range(400))
    tasks2 = list(range(200))
    results1 = charm.pool.map_async(square, tasks1, ncores=2)
    results2 = charm.pool.map_async(add_val, tasks2, ncores=1)
    assert results2.get() == [add_val(x) for x in tasks2]
    assert results1.get() == [square(x) for x in tasks1]
    exit()


charm.start(main)

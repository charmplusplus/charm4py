from charm4py import charm


def square(x):
    return x ** 2


def main(args):
    n = 100000
    CHUNK_SIZE = 16
    # apply function 'square' to elements in 0 to n-1 using the available
    # cores. Parallel tasks are formed by grouping elements into chunks
    # of size CHUNK_SIZE
    result = charm.pool.map(square, range(n), chunksize=CHUNK_SIZE)
    assert result == [square(i) for i in range(n)]
    exit()


charm.start(main)

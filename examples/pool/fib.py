from charm4py import charm

# Recursive Parallel Fibonacci
# NOTE: this example is meant to illustrate the use of charm.pool and nested
# parallelism (creating tasks from tasks). However, it is not efficient,
# because the computation performed by each task is very small.


def fib(n):
    if n < 2:
        return n
    return sum(charm.pool.map(fib, [n-1, n-2], allow_nested=True))


def main(args):
    result = fib(13)
    assert result == 233
    exit()


charm.start(main)

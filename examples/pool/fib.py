from charm4py import charm, coro

# Recursive Parallel Fibonacci
# NOTE: this example is meant to illustrate the use of charm.pool and nested
# parallelism (creating tasks from tasks). However, it is not efficient,
# because the computation performed by each task (grainsize) is too small.

@coro
def fib(n):
    if n < 2:
        return n
    # this will block here for the result of fib(n-1) and fib(n-2),
    # which is why we mark fib as a coroutine
    return sum(charm.pool.map(fib, [n-1, n-2]))


def main(args):
    result = fib(13)
    assert result == 233
    print('fib(13) is', result)
    exit()


charm.start(main)

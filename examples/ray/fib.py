from charm4py import charm, ray
import time

# Recursive Parallel Fibonacci
# NOTE: this example is meant to illustrate the use of charm.pool and nested
# parallelism (creating parallel tasks from other parallel tasks). However, it
# is very inefficient, because the computation performed by each task
# (grainsize) is very small.
#
# See fib-numba.py for a more efficient version where the grainsize can be
# controlled and the sequential computation runs with Numba.

# ray version

@ray.remote
def fib(n):
    if n < 2:
        return n
    else:
        # this will create two tasks which will be sent to distributed workers
        # (tasks can execute on any PE). map will block here for the result of
        # fib(n-1) and fib(n-2), which is why we mark fib as a coroutine
        # return sum(charm.pool.map(fib, [n-1, n-2]))
        result1 = fib.remote(n-1)
        result2 = fib.remote(n-2)
        return ray.get(result1)+ray.get(result2)


def main(args):
    ray.init()
    print('\nUsage: fib.py [n]')
    n = 12
    if len(args) > 1:
        n = int(args[1])
    print('Calculating fibonacci of N=' + str(n))
    t0 = time.time()
    result = fib.remote(n)
    print('Result is', ray.get(result), 'elapsed=', round(time.time() - t0, 3))
    #charm.thisProxy.stop_profiling()
    exit()

charm.start(main)

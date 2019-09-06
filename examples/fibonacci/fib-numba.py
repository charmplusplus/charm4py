from charm4py import charm, coro, Chare, Group
import time
import numba

# Recursive Parallel Fibonacci
# This example is meant to illustrate the use of charm.pool and nested
# parallelism (creating parallel tasks from other parallel tasks).

# NOTE that the grainsize is a critical parameter for performance. The fibonacci
# number is calculated in a parallel recursive manner by spawning parallel
# tasks until n <= GRAINSIZE, at which point fib(n) is calculated on the process
# where the task is running using a sequential algorithm JIT-compiled with Numba.
# - If GRAINSIZE is too low, *many* tasks will be spawned and you will pay the
#   cost of creating them, scheduling and communication.
# - If GRAINSIZE is too high, you might not get a sufficient number of tasks to
#   achieve high parallel efficiency (this depends on how many cores you are
#   running on).


@coro
def fib(n):
    if n <= GRAINSIZE:
        return fib_seq(n)
    else:
        # this will create two tasks which will be sent to distributed workers
        # (tasks can execute on any PE). map will block here for the result of
        # fib(n-1) and fib(n-2), which is why we mark fib as a coroutine
        return sum(charm.pool.map(fib, [n-1, n-2]))


@numba.jit(nopython=True, cache=False)  # numba really speeds up the computation
def fib_seq(n):
    if n < 2:
        return n
    else:
        return fib_seq(n-1) + fib_seq(n-2)


class Util(Chare):
    def compile(self):
        fib_seq(3)


def main(args):
    global GRAINSIZE
    print('\nUsage: fib-numba.py [n] [grainsize]')
    n = 40
    if len(args) > 1:
        n = int(args[1])
    GRAINSIZE = n - 5
    if len(args) > 2:
        GRAINSIZE = int(args[2])
    GRAINSIZE = max(2, GRAINSIZE)
    # set GRAINSIZE as a global variable on all processes before starting
    charm.thisProxy.updateGlobals({'GRAINSIZE': GRAINSIZE}, awaitable=True).get()
    # precompile fib_seq on every process before the actual computation starts,
    # by calling the function. this helps get consistent benchmark results
    Group(Util).compile(awaitable=True).get()
    print('Calculating fibonacci of N=' + str(n) + ', grainsize=', GRAINSIZE)
    t0 = time.time()
    result = fib(n)
    print('Result is', result, 'elapsed=', round(time.time() - t0, 3))
    exit()


charm.start(main)

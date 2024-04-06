from charm4py import charm, coro

# Recursive Parallel Fibonacci

@coro
def fib(n):
    if n < 2:
        return n
    return sum(charm.pool.map(fib, [n-1, n-2]))

def main(args):
    print('fibonacci(13)=', fib(13))
    exit()

charm.start(main)

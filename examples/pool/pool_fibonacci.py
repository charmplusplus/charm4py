from charm4py import charm, coro

# Recursive Parallel Fibonacci

@coro
def fib(n):
    if n < 2:
        return n
    return sum(charm.pool.map(fib, [n-1, n-2]))

def main(args):
    if(charm.numPes()==1):
        print("Error: Run with more than one PE (exiting). For documentation on using pools, see: https://charm4py.readthedocs.io/en/latest/pool.html")
        exit()
    print('fibonacci(13)=', fib(13))
    exit()

charm.start(main)

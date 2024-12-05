from charm4py import charm, Chare, Future, coro
#modeled after the charm with futures example in the charm++ textbook

THRESHOLD = 20

class Fib(Chare):

    @coro
    def __init__(self, n, future):
        if n < THRESHOLD:
            res = self.seqFib(n)
            future.send(res)
        else:
            # Create two futures for the recursive calls
            f1 = Future()
            f2 = Future()

            # Create two new chares with parameters n - 1, n - 2, and their corresponding futures
            childfib1 = Chare(Fib, args=[n - 1, f1])
            childfib2 = Chare(Fib, args=[n - 2, f2])

            # Wait for the results
            val1 = f1.get()
            val2 = f2.get()
            res = val1 + val2

            # Send result back to the parent chare
            future.send(res)

    def seqFib(self, n):
        if n <= 1:
            return n
        else:
            return self.seqFib(n - 1) + self.seqFib(n - 2)

@coro
def main(args):
    if len(args) < 2:
        print("Possible Usage: charmrun ++local +p4 fibonacciWithFutures.py 20 <n>")
        charm.exit()
    n = int(args[1])
    if n < 0:
        print("n must be a non-negative integer")
        charm.exit()

    # Create a future
    f = Future()

    # Create a Fib chare to start the calculations
    fibChare = Chare(Fib, args=[n, f])

    # Get the value of the future (blocks until received)
    res = f.get()
    print("The requested Fibonacci number is:", res)
    charm.exit()

charm.start(main)

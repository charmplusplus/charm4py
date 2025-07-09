====================
:code:`charm4py.ray`
====================


The :code:`charm4py.ray` module is an implementation of the Ray Core API using a Charm4Py
backend. This module is installed as part of the Charm4Py installation.

Short Example
-------------

The following is an example of a Ray program to calculate the nth fibonacci number 
written using :code:`charm4py.ray`,

.. code-block:: python

    from charm4py import charm, ray
    import time

    @ray.remote
    def fib(n):
        if n < 2:
            return n
        else:
            result1 = fib.remote(n-1)
            result2 = fib.remote(n-2)
            return ray.get(result1) + ray.get(result2)


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
        exit()

    charm.start(main)


The only difference between :code:`ray` and :code:`charm4py.ray` is the import statement and
the call to :code:`main` function.
====
Pool
====

.. .. contents::

The Charm Pool is a library on top of Charm4py that can schedule sets of *tasks*
among the available hosts and processors. Tasks can also spawn other tasks.
There is only one pool that is shared among all processes. Any tasks,
regardless of when or where they are spawned, will use the same pool of
distributed workers, thereby avoiding unnecessary costs like process creation
or creating more processes than processors.

.. warning::

    ``charm.pool`` is experimental. It outperforms other Python frameworks,
    particularly when using an efficient communication layer like MPI, but
    the API and performance (especially at large scales) is still subject to
    change.

.. note::

    The current implementation of ``charm.pool`` reserves process 0 for a
    scheduler. This means that if you are running charm4py with N processes,
    there will be N-1 pool workers, and thus N-1 is the maximum speedup using
    the pool.


The pool can be used at any point after the application has started, and can be
used from any process. Note that there is no limit to the amount of "jobs" that
can be sent to the pool at the same time.

The main function of ``charm.pool`` is currently parallel map. The syntax is:

* **charm.pool.map(function, iterable, ncores=-1, chunksize=1, allow_nested=False)**

  This is a parallel equivalent of the map function, which applies ``function`` to
  every item of ``iterable``, returning the results. It divides the iterable into
  a number of chunks, based on the ``chunksize`` parameter, and submits them to the
  process pool, each as a separate task. This method blocks the calling thread
  until the result arrives.

  The parameter ``ncores`` limits the job to use a specified number of cores.
  If this value is negative, the pool will use all available cores (note that
  the total number of available cores is determined at application launch).

  Use ``allow_nested=True`` if you want tasks to be able to spawn other parallel
  work (for example, tasks that themselves call ``charm.pool``).


* **charm.pool.map_async(function, iterable, ncores=-1, chunksize=1, allow_nested=False)**

  This is the same as the previous method but immediately returns a future, which
  can be queried asynchronously (see :ref:`Futures <futures-api-label>` for
  more information).


Examples
--------

.. code-block:: python

    from charm4py import charm

    def square(x):
        return x**2

    def main(args):
        result = charm.pool.map(square, range(10), chunksize=2)
        print(result)  # prints [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        exit()

    charm.start(main)


Note that due to communication and other overheads, grouping items into chunks
(with *chunksize*) is necessary for best efficiency when the duration of
tasks is very small (e.g. less than one millisecond). How small a task size
(aka grain size) the pool can efficiently support depends on the actual overhead,
which depends on communication performance (network speed, communication layer
used -TCP, MPI, etc-, number of hosts...). The chunksize parameter can be used
to automatically increase the grainsize.


.. code-block:: python

    from charm4py import charm

    # Recursive Parallel Fibonacci

    def fib(n):
        if n < 2:
            return n
        return sum(charm.pool.map(fib, [n-1, n-2], allow_nested=True))

    def main(args):
        print('fibonacci(13)=', fib(13))
        exit()

    charm.start(main)

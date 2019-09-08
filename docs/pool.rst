====
Pool
====

.. .. contents::


Pool is a library on top of Charm4py that can schedule sets of "tasks"
among the available hosts and processors. Tasks can also spawn other tasks.
A task is simply a Python function.
There is only one pool that is shared among all processes. Any tasks,
regardless of when or where they are spawned, will use the same pool of
distributed workers, thereby avoiding unnecessary costs like process creation
or creating more processes than processors.

.. warning::

    ``charm.pool`` is experimental,
    the API and performance (especially at large scales) is still subject to
    change.

.. note::

    The current implementation of ``charm.pool`` reserves process 0 for a
    scheduler. This means that if you are running Charm4py with N processes,
    there will be N-1 pool workers, and thus N-1 is the maximum speedup using
    the pool. You might want to adjust the number of processes accordingly.


The pool can be used at any point after the application has started, and can be
used from any process. Note that there is no limit to the amount of "jobs" that
can be sent to the pool at the same time.

The API of ``charm.pool`` is:

* **map(func, iterable, chunksize=1, ncores=-1)**

    This is a parallel equivalent of the map function, which applies the function
    *func* to every item of *iterable*, returning the list of results. It
    divides the iterable into a number of chunks, based on the *chunksize*
    parameter, and submits them to the pool, each as a separate task.
    This method blocks the current coroutine until the result arrives.

    The parameter *ncores* limits the job to use a specified number of cores.
    If this value is negative, the pool will use all available cores (note that
    the total number of available cores is determined at application launch).

    Use the ``@coro`` decorator on your functions if you want them to be able
    to suspend (for example, if they create other tasks and need to wait
    for the results).

* **map_async(func, iterable, chunksize=1, ncores=-1)**

    This is the same as the previous method but immediately returns a
    :ref:`Future <futures-api-label>`, which can be queried asynchronously.

* **Task(func, args, ret=False, awaitable=False)**

    Create a single task to run the function *func*. The function will receive
    *args* as unpacked arguments.

    By default this returns nothing.
    If *awaitable* is ``True``, the call returns a :ref:`Future <futures-api-label>`,
    which can be used to wait for completion of the task.
    If *ret* is ``True``, the call returns a :ref:`Future <futures-api-label>`,
    which can be used to wait for the task's return value.

    Creating a single task is similar to using ``map_async(func, iterable)`` with
    an iterable of length one. There are, however, some subtle differences:

    - By default it doesn't create a future or receive a result, which is less
      expensive.
    - The task can spawn other tasks without having to be a coroutine (if it
      doesn't request a future).
    - The task receives the arguments unpacked.


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

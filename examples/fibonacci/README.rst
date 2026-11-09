
**Recursive Parallel Fibonacci**

These examples calculate the Fibonacci number at a given index provided
by the user.

This example is meant to illustrate the use of ``charm.pool`` and nested
parallelism (creating parallel tasks from other parallel tasks).

**Different versions**

There are 3 implementations:

-  fib.py (Uses the charm pool to create tasks recursively and assign them to workers)
-  fib-numba.py (Uses the Numba JIT compiler for an efficient implementation)
-  fibonacci_with_futures.py (Uses Charm4Py futures to store the results of intermediate calculations)

**Usage**

$ python3 -m charmrun.start +p<N> <file_name> <index>

where N is the number of PEs and index is the Fibonacci number you want to calculate. For example, index=6
should return 8.

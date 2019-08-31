
This example implements a simple distributed job scheduler.

Jobs consist of one or more tasks. The type of job supported in this
example is "parallel map", that applies a function to the elements of an
iterable, using all available processes, returning the list of results.
The scheduler is distributed in the sense that it will send tasks to any
available processes, including processes on remote hosts.
But there is only one central scheduler on process 0. The scheduler keeps
track of which processes are idle (not running any tasks) and sends tasks
to processes as they become idle (thus it balances load in a natural manner).
Multiple jobs can be submitted to the scheduler, at the same or different
times.

**NOTE**: This is not meant to be the most scalable or efficient implementation.
For example, grouping tasks into chunks for better efficiency is not
implemented in this example.
Charm4py has a more advanced implementation of this in ``charm4py/pool.py``.

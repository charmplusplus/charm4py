========
Examples
========

There are several examples in the source code repository, with documentation
and comments: https://github.com/charmplusplus/charm4py/tree/main/examples

These include:

- A simple distributed job/task scheduler.

- Recursive parallel Fibonacci calculator using :doc:`pool` to spawn tasks.
  Includes a Numba accelerated version.

- N-Queen problem parallel solver, using a simple state space search
  algorithm implemented with :doc:`pool` tasks. Includes a Numba accelerated version.

- Jacobi iteration on a 2D array. Can use Numba.

- 2D particle simulation with dynamic load balancing via migratable chares.

- 2D wave simulation displaying a real-time animation. Can use Numba.


**N-Queen parallel solver**

This program solves the N-Queen problem in parallel.
See https://en.wikipedia.org/wiki/Eight_queens_puzzle for more information.
Basically, it does a recursive parallel state space search of the whole tree
of board states, recording the number of solutions found.

The focus of this example is on simplicity rather than raw speed
of *the sequential computation*. The scaling performance, however,
should be good given a suitable grainsize.

Note that the GRAINSIZE parameter is *critical for performance*. The parameter
means that when there are GRAINSIZE rows left to explore in a branch of the
tree, the remainder of the subtree will be explored on that process using a
sequential algorithm:

- If GRAINSIZE is too low, *many* tasks will be spawned and you will pay the
  cost of creating them, scheduling and communication.

- If GRAINSIZE is too high, you might not get a sufficient number of tasks to
  achieve high parallel efficiency (this depends on how many cores you are
  running on).

Also note that different branches of the tree can have different depth and
size, and thus different computational costs, so for best efficiency you will
still want a reasonable number of tasks.

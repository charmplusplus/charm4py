
**2D particle simulation with dynamic load balancing**

This is a simple simulation of particles moving randomly in a 2D space. The
space is given by a box, and particles moving out of the box appear on the
other side of it. The box is decomposed into a 2D grid of cells. Each cell is a
Chare. In each iteration, particles move randomly. Particles that move out of
the boundary of their cell are sent to the corresponding neighboring cell.

To illustrate **the load balancing capabilities of Charm4py**, we try to induce
load imbalance at the beginning of the simulation by concentrating all the
particles near the center of the grid (chares with more particles do more
work). Note that even with these conditions, whether there is load
imbalance and whether anything can be done about it depends on several other
factors (see below for a discussion).

An scenario which should show significant load imbalance and where dynamic
load balancing improves performance is this:

Cell array of size 8 x 4, running on 4 PEs:

To run without load balancing (this or equivalent command)::

    $ python3 -m charmrun.start +p4 particle.py 8 4 10000

To run with dynamic load balancing::

    $ python3 -m charmrun.start +p4 particle.py 8 4 10000 +balancer GreedyRefineLB


NOTE: Load between chares will tend to become uniform as the simulation
progresses, due to the movement of the particles, which means that after a point
load balancing becomes less important.


About load balancing of persistent migratable chares
----------------------------------------------------

Load imbalance is the situation where some processors do more work than others.
Because resources are used inefficiently, the application takes longer to
execute than it could. Situations where you have persistent objects (chares)
with differing loads can lead to load imbalance. Charm++ can balance load
dynamically in these situations by migrating chares between cores (to equalize
processor load). Note that whether there is actually load imbalance and whether
Charm++ can do anything about it depends also on factors other than simply
non-uniform loads between chares, like the number of chares per PE, and where
the overloaded chares happen to be.

For example, if we have only one chare per PE, even if there is imbalance we
can gain nothing by migrating chares. Therefore we need multiple chares per PE,
also known as *overdecomposition*.

Even with overdecomposition, say we launch the simulation with 4 processes,
and we have 8 chares per PE. Now assume initially we have only 4 "heavy" chares,
and the rest have negligible load. If the runtime places each of the heavy ones
on a different process, there won't be any significant load imbalance (note that
when a chare array is created, Charm4py places the chares on processes based
solely on their index). However, if the runtime places 2 heavy chares on one
process, and 2 on another process, leaving the remaining two processes
without heavy chares, we would have significant load imbalance.

In this example, the load imbalance will vary dynamically due to the motion of
the particles, which is why we need to be able to dynamically balance load.

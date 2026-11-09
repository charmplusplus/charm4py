========
Charm4py
========


.. image:: https://github.com/charmplusplus/charm4py/actions/workflows/charm4py.yml/badge.svg?event=push
       :target: https://github.com/charmplusplus/charm4py/actions/workflows/charm4py.yml

.. image:: https://readthedocs.org/projects/charm4py/badge/?version=latest
       :target: https://charm4py.readthedocs.io/

.. image:: https://img.shields.io/pypi/v/charm4py.svg
       :target: https://pypi.python.org/pypi/charm4py/


Charm4py (Charm++ for Python *-formerly CharmPy-*) is a distributed computing and
parallel programming framework for Python, for the productive development of fast,
parallel and scalable applications.
It is built on top of `Charm++`_, a C++ adaptive runtime system that has seen
extensive use in the scientific and high-performance computing (HPC) communities
across many disciplines, and has been used to develop applications that run on a
wide range of devices: from small multi-core devices up to the largest supercomputers.

Please see the Documentation_ for more information.

Short Example
-------------

The following computes Pi in parallel, using any number of machines and processors:

.. code-block:: python

    from charm4py import charm, Chare, Group, Reducer, Future
    from math import pi
    import time

    class Worker(Chare):

        def work(self, n_steps, pi_future):
            h = 1.0 / n_steps
            s = 0.0
            for i in range(self.thisIndex, n_steps, charm.numPes()):
                x = h * (i + 0.5)
                s += 4.0 / (1.0 + x**2)
            # perform a reduction among members of the group, sending the result to the future
            self.reduce(pi_future, s * h, Reducer.sum)

    def main(args):
        n_steps = 1000
        if len(args) > 1:
            n_steps = int(args[1])
        mypi = Future()
        workers = Group(Worker)  # create one instance of Worker on every processor
        t0 = time.time()
        workers.work(n_steps, mypi)  # invoke 'work' method on every worker
        print('Approximated value of pi is:', mypi.get(),  # 'get' blocks until result arrives
              'Error is', abs(mypi.get() - pi), 'Elapsed time=', time.time() - t0)
        exit()

    charm.start(main)


This is a simple example and demonstrates only a few features of Charm4py. Some things to note
from this example:

- *Chares* (pronounced chars) are distributed Python objects.
- A *Group* is a type of distributed collection where one instance of the specified
  chare type is created on each processor.
- Remote method invocation in Charm4py is *asynchronous*.

In this example, there is only one chare per processor, but multiple chares (of the same
or different type) can exist on any given processor, which can bring flexibility and also performance
benefits (like dynamic load balancing). Please refer to the documentation_ for more information.


Contact
-------

We would like feedback from the community. If you have feature suggestions,
support questions or general comments, please visit the repository's `discussion page`_
or email us at <charm@cs.illinois.edu>.

Main author at <jjgalvez@illinois.edu>


.. _Charm++: https://github.com/charmplusplus/charm

.. _Documentation: https://charm4py.readthedocs.io

.. _discussion page: https://github.com/charmplusplus/charm4py/discussions

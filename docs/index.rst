=======
Charmpy
=======

Charmpy is a high-level parallel and distributed programming framework with a simple
and powerful API, based on migratable Python objects and remote method invocation;
built on top of an adaptive C/C++ runtime system providing *speed*, *scalability* and
*dynamic load balancing*.

Charmpy allows writing parallel and distributed applications in Python based on
the `Charm++ programming model`_. Charm++ has seen extensive use in the
scientific and high performance computing (HPC) communities across a wide variety of
computing disciplines, and has been used to produce several large parallel applications
that run on the largest supercomputers, like NAMD_.

With Charmpy, all the application code can be written in Python. The core Charm++
runtime is implemented in a C/C++ shared library which the ``charmpy``
module interfaces with.

As with any Python program, there are several methods available to support
high-performance functions where needed. These include, among others: NumPy_;
writing the desired functions in Python and JIT compiling to native machine
instructions using Numba_; or accessing C or Fortran code using f2py_.
Another option for increased speed is to run the program using a fast Python
implementation (e.g. PyPy_).

We have found that using charmpy + numba, it is possible to build parallel applications
entirely in Python that have the same or similar performance as the equivalent C++ Charm
application, and that scale to hundreds of thousands of cores.

Example applications are in the ``examples`` subdirectory.

.. _numpy: http://www.numpy.org/

.. _Numba: http://numba.pydata.org/

.. _f2py: http://www.f2py.com/

.. _Charm++ programming model: http://charmplusplus.org/

.. _NAMD: http://www.ks.uiuc.edu/Research/namd/

.. _PyPy: http://pypy.org/

.. toctree::
   :maxdepth: 2

   features
   overview
   install
   setup
   running
   tutorial
   performance
   benchmarks
   contact

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

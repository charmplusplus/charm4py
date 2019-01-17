========
charm4py
========

charm4py is a high-level parallel and distributed programming framework with a simple
and powerful API, based on migratable Python objects and remote method invocation;
built on top of an adaptive C/C++ runtime system providing *speed*, *scalability* and
*dynamic load balancing*.

charm4py allows writing parallel and distributed applications in Python based on
the `Charm++ programming model`_. Charm++ has seen extensive use in the
scientific and high performance computing (HPC) communities across a wide variety of
computing disciplines, and has been used to produce several large parallel applications
that run on the largest supercomputers, like NAMD_.

With charm4py, all the application code can be written in Python. The core Charm++
runtime is implemented in a C/C++ shared library which the ``charm4py``
module interfaces with.

As with any Python program, there are several methods available to support
high-performance functions where needed. These include, among others: NumPy_;
writing the desired functions in Python and JIT compiling to native machine
instructions using Numba_; or accessing C or Fortran code using f2py_.
Another option for increased speed is to run the program using a fast Python
implementation (e.g. PyPy_).

We have found that using charm4py + Numba, it is possible to build parallel applications
entirely in Python that have the same or similar performance as the equivalent C++
application (whether based on Charm++ or MPI), and that scale to hundreds of thousands of cores.

Example applications are in the ``examples`` subdirectory of the source code repository_.

.. _numpy: http://www.numpy.org/

.. _Numba: http://numba.pydata.org/

.. _f2py: http://www.f2py.com/

.. _Charm++ programming model: http://charmplusplus.org/

.. _NAMD: http://www.ks.uiuc.edu/Research/namd/

.. _PyPy: http://pypy.org/

.. _repository: https://github.com/UIUC-PPL/charm4py

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   features
   overview
   install
   running
   tutorial

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   core-api
   pool

.. toctree::
   :maxdepth: 2
   :caption: Performance

   performance
   benchmarks

.. toctree::
   :maxdepth: 1
   :caption: Misc

   contact
   release-notes

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

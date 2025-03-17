========
Charm4py
========

Charm4py is a distributed computing and parallel programming framework for
Python, for the productive development of fast, parallel and scalable applications.
It is built on top of `Charm++`_, an adaptive runtime system that has seen
extensive use in the scientific and high-performance computing (HPC) communities
across many disciplines, and has been used to develop applications like NAMD_
that run on a wide range of devices: from small multi-core devices up
to the largest supercomputers.


.. _Charm++: https://charmplusplus.org/

.. _NAMD: https://www.ks.uiuc.edu/Research/namd/



.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   features
   introduction
   install
   running
   tutorial
   examples

.. toctree::
   :maxdepth: 2
   :caption: Manual

   charm-api
   chare-api
   collections-api
   reductions-api
   futures-api
   channels
   sections
   pool
   rules

.. toctree::
   :maxdepth: 2
   :caption: Performance

   perf-tips
   profiling
   serialization

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

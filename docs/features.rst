============
Features
============

.. .. contents::

- **Speed**: Runtime overhead is very low, particularly compared to Python
  libraries with similar features. Charmpy runs on top of `Charm++`_, a C/C++
  runtime system designed to run High-performance computing (HPC) applications
  and to scale to hundreds of thousands of cores.
- Simple and powerful API
- Programming model based on objects and remote method invocation offers easy
  translation path from serial Python programs.
- Asynchronous message passing
- Automatic overlap of communication and computation
- Dynamic load balancing
- High-performance communication supported on many systems (RDMA, CMA, ...)


.. _Charm++: http://charmplusplus.org/


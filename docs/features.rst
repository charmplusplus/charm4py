============
Features
============

- **Actor model**: Charm4py employs a simple and powerful actor model for concurrency and
  parallelism. Applications are composed of distributed Python objects; objects can
  invoke methods of any other objects in the system, including
  those on other hosts. This happens via message passing, and works in the same
  way regardless of the location of source and destination objects.

- **Asynchronous**: every operation, including remote method invocation, is executed asynchronously.
  This contributes to better resource utilization and overlap of computation and communication.

- **Concurrency**: multiple concurrency features are seamlessly integrated into the actor model,
  including coroutines, channels and futures, that facilitate writing in direct or
  sequential style.
  See the :doc:`introduction` for a quick overview.

- **Speed**: The core Charm++ library is implemented in C/C++, making runtime
  overhead very low. Charm4py calls it through one C++ extension module
  (``charm4py._ckdyn``), which builds and delivers messages in C++.
  `Charm++`_ has been used in high-performance computing for many years,
  with applications scaling to the world's top supercomputers.

- **Load balancing** of persistent objects: distributed objects can be migrated
  by the runtime dynamically to balance computational load, in a way that is
  transparent to applications.

- **Parallel tasks** using a distributed pool of workers (which works across
  multiple hosts). Tasks are Python functions and coroutines. The framework supports
  efficient nested parallelism (tasks can create and wait for other tasks). Among
  the operations supported are large-scale parallel map (akin to Python multiprocessing's map),
  and the ability to spawn individual tasks, which can be used to easily implement
  parallel state space search or similar algorithms. The runtime decides where
  to launch tasks and balances them across processes.

- **High-performance communication**: Charm4py runs on Reconverse, whose
  communication layer LCI uses libfabric or InfiniBand verbs, so a source build
  against a site's libfabric uses the cluster's high-speed network (see
  :doc:`install`). Wheels include libfabric's TCP providers (and shared memory
  on Linux) only.



.. _Charm++: https://charmplusplus.org/

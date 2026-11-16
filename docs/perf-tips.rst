================
Performance Tips
================

Charm4py will help you parallelize and scale your applications, but
it won't make the sequential parts of your code faster.
For this, there are several technologies that accelerate Python code, like
NumPy_, Numba_, and Cython_. These are outside the scope of this section,
but we highly recommended using Numba. We have found that using Charm4py + Numba,
it is possible to build parallel applications entirely in Python that have the
same or similar performance as the equivalent C++ application. Many examples
in our source code repository use Numba.


This section contains tips to help maximize the performance of your applications
by reducing runtime overhead. Overhead becomes apparent at very low
method or task granularity and high communication frequency. Therefore, whether these
tips actually help depends on the nature of your
application and the impact overhead has on it. Also keep in mind that there
are other factors besides overhead that can affect performance, and are outside
the scope of this section.

.. note::
  Method granularity refers to the time for a chare's remote method to run or, in the
  case of coroutines, the time the method runs before it suspends and
  control is switched to a different task.


- For communication between hosts on a cluster with a high-speed network
  (InfiniBand, Slingshot, Omni-Path), install Charm4py from source against the
  site's libfabric or libibverbs. A wheel includes a libfabric with TCP providers
  only, and shared memory on Linux (see :doc:`install`).

.. - Coroutines are very lightweight, but do add a tiny bit of overhead. For
..   very small methods that do a negligible amount of work but are called frequently,
..   you might want to consider avoiding use of coroutines (rely just on message
..   passing and method invocation).

- If you are sending large arrays of data, use Numpy arrays (or arrays from Python's
  ``array`` package) and send each as a separate parameter.
  This allows Charm4py to directly
  copy the contents of the arrays to a message that is sent through the
  network (thus bypassing pickling/serialization libraries). For example:
  ``proxy.method(array1, array2, array3)``.

  In the case of updateGlobals, have each array be an element of the dict,
  for example: ``charm.thisProxy.updateGlobals({'array1': array1, 'array2': array2, ...})``

  With channels, do the following: ``ch.send(array1, array2, ...)``

  Note that these types of arguments can be freely intermixed with others not
  supporting the buffer protocol.

- If you are frequently indexing a proxy (for example ``myproxy[3]``) it is more
  efficient to store the proxy to the individual element and reuse it, for example::

    elem_proxy = myproxy[3]
    for _ in range(100):
      elem_proxy.work(...)

- When calling remote methods, it is generally more efficient to use unnamed arguments.

- Avoiding ``awaitable=True`` and ``ret=True`` in the critical path can reduce
  overhead in some cases. Internally, awaitable calls require creating a future
  and sending it as part of your remote method call. It should always be
  possible to rewrite code so that notification of completion or results are
  sent via a separate and explicit method invocation, although this can tend to
  result in less readable code.

- Make sure profiling is disabled (it is disabled by default). Charm4py prints
  a warning at startup if it is enabled.

- Large numpy arrays in point-to-point sends go through Charm++ zero-copy
  (``charm.options.zerocopy_threshold``, 1 MB by default), and received arrays
  are views of the message, not copies. Do not set
  ``charm.options.copy_received_arrays`` unless the program needs it, and copy
  only the received arrays it keeps or modifies (see :doc:`charm-api`).

.. _numpy: https://www.numpy.org/

.. _Numba: https://numba.pydata.org/

.. _Cython: https://cython.org/


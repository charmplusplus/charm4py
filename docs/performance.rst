============
Performance
============

.. contents::

Python 3 with the Cython interface layer is currently the recommended way to run
Charm4py programs to get the best performance (on CPython). This is the option that
is installed when using pip.

Performance analysis
--------------------

Set ``charm4py.Options.PROFILING`` to ``True`` to activate profiling. Do this this
before the program starts.

``charm.printStats()``: prints timing results and message statistics
*for the processor where it is called*.
A good place to use this is typically right before exiting the program.

Here is some example output of ``printStats()`` for ``examples/particle/particle.py``
executed on 4 PEs:

.. code-block:: text

    Timings for PE 0:
    |                   | em    | send  | recv  | total |
    ------ <class '__main__.Main'> as Mainchare ------
    | collectMax        | 0.003 | 0.0   | 0.002 | 0.005 |
    | done              | 0.0   | 0.0   | 0.0   | 0.0   |
    ------ <class '__main__.Cell'> as Array ------
    | getNbIndexes      | 0.0   | 0.0   | 0.0   | 0.0   |
    | resumeFromSync    | 0.518 | 0.009 | 0.001 | 0.528 |
    | run               | 0.028 | 0.001 | 0.0   | 0.029 |
    | updateNeighbor    | 4.942 | 0.113 | 0.069 | 5.124 |
    -----------------------------------------------------
    |                   | 5.491 | 0.123 | 0.072 | 5.686 |
    -----------------------------------------------------
    | reductions        |       |       | 0.002 | 0.002 |
    | custom reductions |       |       | 0.0   | 0.0   |
    | migrating out     |       |       | 0.0   | 0.0   |
    -----------------------------------------------------
    |                   | 5.491 | 0.123 | 0.074 | 5.689 |

    Messages sent: 4556
    Message size in bytes (min / mean / max): ['0', '395.842405618964', '3121']
    Total bytes = 1.72 MB

    Messages received: 4110
    Message size in bytes (min / mean / max): ['0', '461.45182481751823', '3713']
    Total bytes = 1.809 MB

The output first shows timings (in seconds) for each chare that was active on the PE.
Each row corresponds to a different entry method, where "entry" refers to the method
being used as the target of a remote method invocation (via a proxy).
Note that if the entry method itself calls other functions and methods locally
(without a proxy), the time to execute those will be added to its own timings.

Timings are shown in four colums:

:em: Time in user code (outside of runtime) executing an entry method.
:send: Time in proxy and contribute calls (in runtime).
:recv: Time between Charm4py layer receiving a message for delivery and the target
  entry method being invoked (in runtime).
:total: Sum of the previous three columns.

The last rows show miscellanous overheads pertaining to reductions and migration.

.. note::
    While all of the Charm4py code is instrumented, there are parts of the C/C++
    runtime that are not currently reflected in the above timings.

The last part of the output shows message statistics for remote method invocations (number
of messages sent and received and their sizes).

Charm++ has powerful tracing functionality and a performance analysis and visualization
tool called *Projections*. This functionality has not yet been integrated into Charm4py.

.. _perf-serialization-label:

Serialization
-------------

In many cases a remote method invocation results in serialization of the arguments
into a message that is sent to a remote process.
Serialization is also referred to as *pickling*. Pickling can account for much of
the overhead of the Charm4py runtime. Fastest
serialization is obtained with the C implementation of the ``pickle`` module
(only available in CPython).

.. important::
    Pickling can be bypassed for certain data and is encouraged for best performance
    (see next subsection).

A general guideline to achieve good performance is to avoid passing custom types as
arguments to remote methods in *the application's critical path*.
Examples of recommended types to use for best performance include: Python containers
(lists, dicts, set), basic datatypes (int, float, str, bytes) or combinations of the
above (e.g.: dict containing lists of ints). Custom objects are automatically
pickled but can significantly affect the performance of pickle and therefore their
use inside the critical path is not recommended.

Bypassing pickling
~~~~~~~~~~~~~~~~~~

**This feature is currently only fully supported with Python 3 and Cython/CFFI**.

Best performance is achieved when passing arguments that support the buffer protocol
(`byte arrays`_, array.array_ and `NumPy arrays`_). These bypass pickling altogether and
are directly copied from their memory buffer in Python into a message in the Charm
C++ library for sending. Note that these types of arguments can be freely intermixed
with others not supporting the buffer protocol. For example:

.. code-block:: python

    particle1 = Particle(3, 5)    # Particle is a user-defined custom type
    particle2 = Particle(7, 19)
    A = numpy.arange(100)         # 100 element numpy array
    proxy.work([1,2], particle1, A, particle2) # arguments 0, 1 and 3 will be pickled,
                                               # 2 will bypass pickling


.. _byte arrays: https://docs.python.org/3/library/stdtypes.html#bytes

.. _array.array: https://docs.python.org/3/library/array.html

.. _NumPy arrays: https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html

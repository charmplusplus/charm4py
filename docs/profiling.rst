=========
Profiling
=========

Charm4py provides basic functionality for profiling. Please note that it may
not be suitable for full-blown parallel performance analysis. Essentially, it
measures the time spent inside the Charm4py runtime and application, and
presents the information broken down by remote methods, and send and receive
overhead.

.. warning::

  The profiling tool:

  - Does not track time inside the Charm++ scheduler, or time waiting for
    messages to arrive from the network.

  - The tool provides separate timings for each PE. Timings for one PE can be
    very different from those of another PE (depending on the application), so
    you should print the information for all the PEs you are interested in.

  - Currently the tool does not provide timings for a specific time period,
    so it can be hard to determine how much work each PE is doing in a specific
    time interval.

It is important to note that the total application time could be larger than
the total time reported for one PE by the profiling tool. This can happen, for
example, if the parallel efficiency is not good (e.g. idle processes waiting
for work to complete in other processes, load imbalance, ...), or processes spending a
lot of time waiting for communication to complete (which depends on the size
and number of messages per second, network throughput and latency, efficiency
of the communication layer, etc.). If you are seeing significant idle time, one
thing that can increase performance is using an efficient communication layer
such as MPI (see :doc:`perf-tips`).

In the future, we plan to support Projections_ which is a full-fledged
parallel performance analysis tool for Charm++.


Usage
-----

Set ``charm.options.profiling`` to ``True`` before the runtime starts
in order to activate profiling.

To print profiling results at any time, call ``charm.printStats()``.
This prints timings and message statistics *for the PE where it is called*.
A good place to use this is right before exiting the application.
You can also invoke this remotely, doing ``charm.thisProxy[pe].printStats()``
(like any other remote method, you can also wait for its completion).

Here is some example output of the profiler from ``examples/particle/particle.py``
executed on 4 PEs with ``python3 -m charmrun.start +p4 particle.py +balancer GreedyRefineLB``:

.. code-block:: text

    Timings for PE 0 :
    |                               | em    | send  | recv  | total |
    ------ <class '__main__.Cell'> as Array ------
    | __init__                      | 0.001 | 0.001 | 0.0   | 0.002 |
    | migrated                      | 0.0   | 0.0   | 0.027 | 0.028 |
    | AtSync                        | 0.0   | 0.0   | 0.0   | 0.0   |
    | _channelConnect__             | 0.001 | 0.0   | 0.001 | 0.002 |
    | _channelRecv__                | 0.037 | 0.0   | 0.056 | 0.093 |
    | _coll_future_deposit_result   | 0.0   | 0.0   | 0.0   | 0.0   |
    | _getSectionLocations_         | 0.0   | 0.0   | 0.0   | 0.0   |
    | getInitialNumParticles        | 0.0   | 0.0   | 0.0   | 0.0   |
    | getNbIndexes                  | 0.0   | 0.0   | 0.0   | 0.0   |
    | getNumParticles               | 0.0   | 0.0   | 0.0   | 0.0   |
    | migrate                       | 0.0   | 0.0   | 0.0   | 0.0   |
    | reportMax                     | 0.0   | 0.0   | 0.0   | 0.0   |
    | resumeFromSync                | 0.0   | 0.0   | 0.0   | 0.001 |
    | run                           | 3.247 | 0.056 | 0.001 | 3.304 |
    | setMigratable                 | 0.0   | 0.0   | 0.0   | 0.0   |
    -----------------------------------------------------------
    |                               | 3.288 | 0.058 | 0.086 | 3.432 |
    -----------------------------------------------------------
    | reductions                    |       |       | 0.0   | 0.0   |
    | custom reductions             |       |       | 0.0   | 0.0   |
    | migrating out                 |       |       | 0.002 | 0.002 |
    -----------------------------------------------------------
    |                               | 3.288 | 0.058 | 0.088 | 3.434 |

    Messages sent: 6957
    Message size in bytes (min / mean / max): 0 / 234.319 / 4083
    Total bytes = 1.555 MB

    Messages received: 6925
    Message size in bytes (min / mean / max): 6 / 291.771 / 315390
    Total bytes = 1.927 MB


The output first shows timings (in seconds) for each chare *type* that was active
on the PE. Each row corresponds to a different entry method (i.e. a remote method
invoked via a proxy).

.. important::
    Only remote method invocations are measured (not regular function calls).
    So, if a remote method calls other functions and methods locally (without
    using a proxy), the time to execute those will be added to its own time.


Timings are shown in four colums:

:em: Time at the application level (outside of runtime) executing an entry method.
:send: Time in send calls (like proxy calls, reduce, channel send, etc.)
:recv: Time between Charm4py layer receiving a message for delivery and the target
  entry method being invoked.
:total: Sum of the previous three columns.

The last rows show miscellaneous overheads pertaining to reductions and migration.
The last part of the output shows message statistics for remote method invocations (number
of messages sent and received and their sizes).


In this example we can see that most of the time is spent inside the "run"
method of each Cell. Most of the send overhead is from the "run" method
(it calls channel.send() repeatedly) and most of the receive overhead is in the Cell's
``_channelRecv__`` method (which is a method of the parent class ``Chare`` that
is called when one of its channels receives a message). There is also some
receive overhead due to chares migrating into this PE (method ``migrated``).



.. _Projections: https://charm.readthedocs.io/en/latest/projections/manual.html

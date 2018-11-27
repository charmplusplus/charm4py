============
Overview
============

.. .. contents::

*Chares* are distributed Python objects that live and perform work on a processor [#]_.
They can be migrated to different processors by the runtime to, for example,
dynamically balance load. Many objects of the same or different types can live on
one processor. Chares communicate and coordinate between themselves by invoking
their methods, which are defined and called as regular Python methods. This works
regardless of the location of the caller and callee, and the runtime automatically
takes care of location management and of using the most efficient technique for method
invocation. For example, if the communicating objects exist on different hosts,
Charm will automatically pack the method arguments into a message and send the message
to the destination object by the most efficient way possible. [#]_

It is generally desirable to have many objects per core so that the runtime has freedom
to dynamically balance load and to overlap communication and computation, thus
increasing resource utilization and minimizing idle times [#]_.

.. [#] We also refer to processors as cores.

.. [#] Charm4py supports reference passing for objects in the same process,
       Cross Memory Attach for objects in the same host, RDMA for network communications,
       collectives optimized for network topology, among other performance features.
..       Please refer to section TODO for an explanation of these performance features.

.. [#] The Charm runtime automatically schedules messages, work and dynamically
       balances load by migrating objects between cores.
..       More information is in section TODO of the manual

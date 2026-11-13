===============================
Migrating from Charm4py 1.x
===============================

.. .. contents::

What changed and why
--------------------

Charm4py 2 keeps the Python API of 1.x and replaces what is underneath it. In
1.x, a large part of the runtime was written in Python: message dispatch,
argument packing, sections and the local-message path. Charm4py 2 is a thin
binding (the extension module ``charm4py._ckdyn``) over CkDyn, the Charm++
runtime's layer for chares whose classes and entry methods are registered at
run time. Messages, sections, reductions, migration and checkpoints are
handled by Charm++ itself. What this gives a program:

* lower cost per message: on one PE a one-way message costs 0.38 us
  (2.27 us in 1.x), a message to a coroutine 1.48 us (6.14 us), and a 1 MB
  numpy array 24.9 us (59.6 us) (``docs/design/DESIGN.md``, 7.1 step 4a);
* numpy arrays are received as views of the message instead of copies, and
  large arrays sent point to point travel through Charm++ zero-copy;
* GPU (CUDA) arrays can be entry-method arguments in a CUDA build;
* checkpoint and restart (``charm.checkpoint`` and ``+restart``);
* sections are Charm++ ``CkMulticast`` sections, and their members may
  migrate;
* Projections and Summary traces name every Python chare class and entry
  method (``module.Class.method``, ``Class.method`` for a class defined in the
  main script; the group, single-chare and nodegroup registrations of a class
  add ``/group``, ``/chare`` and ``/nodegroup`` to the class name), in a
  Charm++ built with tracing;
* one Charm++ build (Reconverse) for laptops and clusters, shipped inside
  the wheel.

Installing and running
----------------------

Install from a wheel (``pip install charm4py``), from source (``pip install
.``, which fetches and builds Charm++ itself), or against a Charm++ tree you
build yourself. :doc:`install` describes the three ways and the environment
variables. Things that differ from 1.x:

* Charm4py 2 uses only the Reconverse runtime. There is no classic Converse
  build (``netlrts``, ``multicore``, ``mpi``) and no ``charmrun`` binary.
* The Charm++ library must be built with ``--with-refnum-type="unsigned int"``
  (the source builds do this); the extension module refuses a ``libcharm``
  built without it at import.
* Wheels exist for CPython 3.11 to 3.13 on Linux x86_64 and aarch64 and on
  macOS arm64. A wheel uses TCP-based networking only; on a cluster with a
  high-speed network, install from source against the site's libfabric.

**One PE per process.** Charm4py 2 runs one PE in each process. ``+pe N`` is
the total number of PEs of the job, which is also the number of processes.
To run on several PEs, start several processes:

* the Charm4py launcher, which starts the processes on this machine::

    $ python -m charmrun.start +p4 myprogram.py

* LCI's ``lcrun`` (in ``_deps/lci-src/lcrun`` of a Charm++ build
  directory)::

    $ lcrun -n 4 python myprogram.py +pe 4

* the cluster's launcher, one process per PE::

    $ srun -n 128 python myprogram.py +pe 128

``+pe N`` given to a single process stops the program at start-up with::

    charm4py runs one PE per process for now, and this process has N PEs.
    Use several processes with one PE each instead of +pe N in one process:
    lcrun -n N python app.py +pe N, or python -m charmrun.start +pN app.py.

In 1.x on classic Converse, ``+pN ++ppn M`` started processes with ``M``
worker PEs each and one communication thread. There is no equivalent in 2.0:
``++ppn`` has no meaning, and the launcher's ``++nodelist`` (ssh start-up of
remote hosts) is gone; use ``lcrun`` or the cluster's launcher on several
hosts. See :doc:`running`.

Code that runs unchanged
------------------------

The 1.x test suite and the examples run unchanged on Charm4py 2, except for
the items in the next section. This covers chares, groups, arrays, array
maps, proxies, futures, coroutines, ``@when``, channels, reductions
(built-in and custom), callbacks, quiescence detection, readonlies, the pool,
the Ray API, interactive mode, liveViz, steering and appspector. Internals
that some programs read keep their meaning: ``self._numthreads``,
``charm.groups``, ``charm.arrays``, ``charm.entryMethods``,
``charm4py.threads.Future``, ``charm4py.wait.parse_cond_str`` and
``charm.getTopoSubtrees``. ``Chare(C, onPE=p)`` still creates a one-element
array, as in 1.x.

Behaviour changes
-----------------

Received numpy arrays are views, and some are read-only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*What you will see.* A numpy array received from a broadcast, a section
multicast or an ``allreduce`` is read-only. Writing to it raises
``ValueError: assignment destination is read-only``; passing it to
``torch.from_numpy`` prints torch's warning that the NumPy array is not
writable (``examples/mnist/mnist.py`` does this with an ``allreduce``
result).

*Why.* 1.x copied every received array. Charm4py 2 hands the entry method a
numpy view of the message itself (no copy). A broadcast message is shared by
every recipient on the process, so its views are read-only. The result of an
``allreduce`` arrives by broadcast and is read-only too. Arrays from
point-to-point sends, and reduction results delivered to a future, are
writable views.

*What to change.* Copy the array before modifying it
(``a = a.copy()``), or set ``charm.options.copy_received_arrays = True``
before ``charm.start()``, which restores the 1.x behaviour (every received
array is a copy) for the whole program.

A received view keeps its whole message in memory for as long as the array
lives. A program that stores many small received arrays, or a small array
from a message that also carried large ones, holds those messages; copy what
you keep, or use ``copy_received_arrays``.

Non-contiguous numpy arrays are now accepted as arguments (they are made
contiguous before sending).

Arguments are never shared by reference, and are converted one by one
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In 1.x with ``local_msg_optim`` enabled, a call to a chare in the same process
passed its arguments by reference, and :doc:`rules` describes recipients that
share objects with the sender. In 2.0 every call goes through a Charm++
message: the receiver never sees the sender's objects.

Arguments are also converted one by one: numbers, strings, bytes, numpy
arrays, futures and proxies go into the message directly, and everything else
is pickled together. A consequence is that ``proxy.m(a, a)`` with a numpy
array ``a`` arrives as two separate arrays, and an array passed both as an
argument and inside a list arrives as two objects. In 1.x all arguments were
one pickle and arrived as one object. If the receiver relies on identity,
pass the object once.

Numeric reduction results keep the contributors' array shape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*What you will see.* A built-in reduction (``Reducer.sum``, ``max``, ``min``,
``product``, the ``logical_*`` reducers) of numpy arrays with two or more
dimensions returns an array of the contributors' shape: every chare
contributes a ``(2, 3)`` array and the result is a ``(2, 3)`` array. 1.x
returned the flat ``(6,)`` array for a reduction over a whole array or group
and the ``(2, 3)`` array for a section reduction; Charm4py 2 returns
``(2, 3)`` for both. Code that called ``reshape`` on a whole-collection
result keeps working; code that indexed it as a flat array needs
``result.ravel()``.

The shape travels with the contribution (CkDyn's
``CkDynContributeShaped``), so every contributor of one reduction must
contribute the same shape: contributions of two shapes abort the program
with ``CkDyn: contributions to one reduction have different shapes, (2, 3)
and (3, 2)``. For the same reason one reduction cannot mix arrays of two or
more dimensions with 1-D arrays or scalars under a built-in reducer: 1.x
summed a ``(2, 3)`` and a ``(6,)`` contribution as flat data, Charm4py 2
aborts with ``Reduction: contributions to one reduction use different
reducers (CkDyn::shaped (arrays of ndim >= 2) ... and CkReduction::sum_double
...)``. Contribute ``a.ravel()`` everywhere to keep the flat result. A result
with two or more dimensions is an array even with one
element (a ``(1, 1)`` contribution gives a ``(1, 1)`` result); a scalar or a
1-D array of one element still gives a Python scalar, as in 1.x.

Options that are accepted and ignored
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``charm.options.local_msg_optim`` and ``charm.options.local_msg_buf_size``
  are accepted and have no effect. Setting ``local_msg_optim = True`` prints
  once::

    Charm4py> charm.options.local_msg_optim is ignored (local messages always go through Charm++)

  The local-message path stored Python objects by reference and broke
  Charm++ semantics in some cases; it was off by default in 1.x.
  Remove both settings.
* The command-line option ``+libcharm_interface <name>`` is accepted and
  ignored: only one interface exists.
* ``charm.options.pickle_protocol`` defaults to 5 (it was -1, the highest,
  which is 5 on every supported Python). Out-of-band buffers need protocol 5;
  a lower value still works and turns that path off.

``migrate()`` must be the last call of the entry method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``self.migrate(pe)`` packs the element and removes it from this PE before it
returns, as in C++. Code after the call runs on a Python object that has
left: its changes are lost, and a ``contribute`` from it aborts with
``CkDyn: contribute from an object that is not local``. This includes the
reduction of an ``awaitable=True`` broadcast to an entry method that migrates
its element. Call ``migrate`` last, and report completion from
``migrated()``, which runs on the new PE.

An attribute that cannot be pickled stops the program with
``Charm4PyError: cannot pickle <Class><index> for migration: <error>``.
Drop such attributes in ``__getstate__`` and recreate them in
``__setstate__``.

Suspended coroutines and migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A suspended coroutine (waiting on a future, a channel, ``charm.wait`` or
similar) cannot move to another process. While any coroutine of an element is
suspended, the element is marked non-migratable, and load balancers leave it
where it is. ``self.migrate(pe)`` on such an element raises::

    Charm4PyError: Cannot migrate <Class><index>: it has N suspended coroutine(s)

A coroutine that runs to completion does not affect migratability. If you
call ``setMigratable(False)`` yourself, that setting is kept.
(``docs/design/DESIGN.md``, 4.5.)

The AtSync pattern
~~~~~~~~~~~~~~~~~~

Call ``self.AtSync()`` and return; continue in ``resumeFromSync``. The
runtime calls ``resumeFromSync`` through a message, never from inside
``AtSync``, so a ``resumeFromSync`` that calls ``AtSync`` again does not nest.
A coroutine that calls ``AtSync()`` and then waits for something stays
suspended, so its element is not migrated in that balancing step. Every array
created with ``useAtSync=True`` takes part in the PE's barrier, as in Charm++:
a program with two such arrays must call ``AtSync`` on both.

Sections
~~~~~~~~

* ``charm.split``, ``charm.combine``, proxy slicing (``g[::2]``,
  ``a[0:8:2, 1:8:2]``) and ``Group(C, onPEs=[...])`` must be called from a
  coroutine (the main function or a ``@coro`` entry method). Elsewhere they
  raise ``Charm4PyError: charm.split must be called from a coroutine: it
  waits until every member has joined the new section(s)`` (or the same text
  for ``charm.combine``). They return when every member has joined, so the
  returned proxies can be used at once.
* ``charm.combine`` accepts arrays and array sections, including combined
  sections, of one chare class with one number of index dimensions. Groups,
  group sections, or arrays of different classes or dimension counts raise
  ``TypeError`` (for example ``charm.combine accepts array sections and arrays
  only, got ... (group sections cannot be combined)``). 1.x also combined
  group sections and different chare types. An element in several of the
  combined parts is a member once: it receives one copy of each broadcast and
  contributes once.
* A split that leaves a section without members raises
  ``Charm4PyError: charm.split: section <n> has no members``.
* ``proxy.section`` is ``(root_pe, sid)``, where ``sid`` is a 16-byte
  ``bytes`` object; it can be used as a dictionary key and as the ``section=``
  argument of ``contribute``, ``reduce`` and ``allreduce``.
* Section members may migrate (1.x aborted).
* In a constrained group (``Group(C, onPEs=...)``), ``thisProxy`` is the
  plain group proxy while the constructor runs and becomes the section proxy
  right after it.

See :doc:`sections`.

Reserved names
~~~~~~~~~~~~~~

``_sectionVote_`` is a new reserved method of ``Chare``; a class that defines
it fails registration with ``Charm4PyError: <class> redefines reserved method
'_sectionVote_'``. ``__addLocal__``, ``__removeLocal__`` and
``_getSectionLocations_`` are no longer reserved. Charm4py 2 also stores these
attributes on chare objects; do not use the names: ``_ckdyn_target``,
``_ckdyn_packed``, ``_nsuspended`` and ``_sections``. The keyword arguments
``ret``, ``awaitable`` and ``is_ray`` of proxy calls stay reserved, as in 1.x.

Constructors of derived chare classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a chare class that inherits its constructor from another chare class,
1.x ran the parent's constructor once without arguments before the user
constructor. Charm4py 2 does not.

Profiling and appspector
~~~~~~~~~~~~~~~~~~~~~~~~

``charm.printStats()`` no longer has the "send" column: sends are made in
C++ and are not timed. In appspector, a future deposit from another PE is
charged to the coroutine it resumes (1.x charged it to the ``CharmRemote``
group).

New in 2.0
----------

* **Checkpoint and restart.** ``charm.checkpoint(dirname, resume=None)``
  writes all chares, readonlies and runtime state to ``dirname``; running the
  program again with ``+restart dirname`` restores them instead of running
  the main chare's constructor. ``resume`` is an entry method without
  parameters of a group, an array element or the main chare
  (``self.thisProxy.resumed``); the runtime calls it when the checkpoint is
  written and, in the restarted run, once the restart is complete. Program
  flow continues there, because coroutines (including a coroutine ``main``)
  and futures are not restored. See :doc:`charm-api` and
  ``tests/thin/checkpoint_restart.py``.
* **Zero-copy options.** ``charm.options.zerocopy_threshold`` (bytes,
  default 1 MB): a contiguous numpy array of at least this size in a
  point-to-point send goes through Charm++ zero-copy. Broadcasts, section
  multicasts, reduction contributions and pool or Ray task arguments are
  copied into the message whatever their size. A point-to-point call cannot
  pass both a zero-copy host array and a device array (``TypeError``; see
  :doc:`gpus`): send them in two calls or pass one of them by copy.
  ``charm.options.copy_received_arrays`` (default ``False``): see above.

  The receiver reads a zero-copy array from the sender's memory after the
  send call has returned, whereas 1.x copied it at the send. So that a write
  to the array in that interval cannot change what the receiver gets, the
  send makes the array read-only (``a.flags.writeable`` is ``False``) and the
  runtime makes it writeable again when the transfer is done. A program that
  overwrites its send buffer in place right after the send (a ghost exchange
  that refills the same array) then gets ``ValueError: assignment destination
  is read-only`` instead of sending wrong data; fill a new array, send a copy
  (``a.copy()``), or wait until the receiver has replied. An array that was
  read-only before the send stays read-only. Other views of the same memory
  (``a[:]``, ``a.reshape(...)``, the array ``a`` was sliced from) stay
  writeable and are not protected.
* **Device arrays as arguments.** In a Charm++ and Charm4py built with CUDA,
  an object with ``__cuda_array_interface__`` (cupy, torch, numba) passed to
  an entry method in a point-to-point send travels device to device. The
  receiver gets a ``charm4py._ckdyn.DeviceBuffer``, which ``cupy.asarray`` and
  ``torch.as_tensor`` accept without a copy. ``charm.hapiAddCudaCallback``
  keeps its 1.x signature. In a broadcast, a section multicast, a reduction or
  a pool or Ray task, a device array is pickled as in 1.x. Without a device
  build, a device argument raises ``TypeError: device arrays need a CUDA or
  HIP build of Charm++``. Wheels have no device support; see :doc:`gpus` and
  :doc:`install`.
* **Sections**: combined sections are deduplicated unions, members migrate,
  and the section id is usable as a key (see above).
* **NodeGroup.** ``NodeGroup(C, args=[...])`` creates one branch per process,
  indexed by ``charm.myNode()`` (new, with ``charm.numNodes()``), with sends,
  broadcasts, ``ret=True`` gathers, reductions and checkpoint as for groups. See
  :ref:`nodegroup-api-label`.
* **Entry-method attributes.** ``@entry(expedited=True)``, ``inline``,
  ``notrace``, ``appwork``, and for nodegroups ``exclusive`` and ``immediate``
  set the Charm++ attributes of a method. See :ref:`entry-attributes-label`.
* **cProfile per PE.** ``charm.options.cprofiling = True`` profiles every PE
  with ``cProfile`` from start-up to exit and writes
  ``<charm.options.cprofile_filename><pe>.prof`` (default ``profile0.prof``,
  ...); ``charm.exit()`` is unchanged. See :doc:`profiling`.
* **Message priorities.** ``proxy.method(..., prio=p)`` sends with integer
  priority ``p`` (lower runs first), on point-to-point sends and broadcasts; not
  on section sends.
* **Destroying array elements.** ``proxy[i].ckDestroy()`` destroys element
  ``i`` wherever it lives, ``proxy.ckDestroy()`` every element, and
  ``self.ckDestroy()`` the calling element (1.x had none). See
  :ref:`array-api-label` and ``tests/thin/ckdestroy.py``.

Removed or not yet available
----------------------------

* Processes with several PEs (``+pe N`` in one process): refused at
  start-up. Planned for a free-threaded Python.
* Classic Converse builds (``netlrts`` and the others), ``charmrun``'s
  ``++ppn`` and ``++nodelist``.
* Sections are not part of a checkpoint: after ``+restart`` the program
  creates them again. A section id from before the restart raises an error
  when used (the runtime detects it). A constrained group
  (``Group(C, onPEs=...)``) is built on a section, so its ``thisProxy`` does
  not survive a restart either: a program that checkpoints constrained
  groups must recreate them after ``+restart``.
* ``charm.combine`` of groups, group sections or arrays of different classes
  or dimension counts.
* A per-send ``expedited=`` keyword (use ``@entry(expedited=True)`` on the
  method), ``prio=`` on section sends, and priorities on reduction callbacks.
* Wheels have no tracing: Projections and Summary traces need a source build
  with ``CHARM_EXTRA_BUILD_OPTS=--enable-tracing``, whose ``libcharm`` writes
  trace files on every run.
* No CUDA or HIP wheels; device support needs a source build.
* No free-threaded (``3.13t``) wheels and no macOS x86_64 build.
* Examples that import numba (``fib-numba.py``, ``nqueen-numba.py``,
  ``wave2d.py``, LeanMD) need numba installed; ``mnist-mpi4py.py`` needs
  mpi4py.

Checklist
---------

1. Install Charm4py 2 (:doc:`install`); unset ``CHARM4PY_CHARM_DIR`` or point
   it at a Charm++ tree with CkDyn, not a classic build.
2. Run with one PE per process: ``python -m charmrun.start +pN``, ``lcrun -n N
   ... +pe N`` or ``srun -n N ... +pe N``; drop ``++ppn`` and ``++nodelist``.
3. Remove ``local_msg_optim`` and ``local_msg_buf_size`` settings.
4. Find writes to arrays received from broadcasts, multicasts and
   ``allreduce``; copy them, or set ``copy_received_arrays = True``.
5. Check code that relies on two arguments being the same object.
6. Make ``migrate()`` the last call of its entry method; report from
   ``migrated()``.
7. Move work after ``AtSync()`` into ``resumeFromSync``.
8. Call ``split``, ``combine``, slicing and constrained ``Group`` creation
   from a coroutine; restrict ``combine`` to arrays of one class.
9. Rename any chare method ``_sectionVote_`` and attributes ``_ckdyn_*``,
   ``_nsuspended``, ``_sections``.
10. Run the program at 1, 2 and 4 processes.

.. To be confirmed (for the editor; not rendered):
   1. Whether ``++ppn`` given to the launcher or to the program is refused,
      ignored, or passed through to the program's argv.
   2. A future created before a restart: DESIGN 4.5 item 5 says a deposit or
      get() raises Charm4PyError("future from before the restart"); the code
      on thin has no such message. Which one holds?
   3. Restart with a different process count than the checkpointing run
      (tests use the same count).
   4. docs/gpus.rst's closing note says the device paths have not been run on a
      GPU; DESIGN 7.1 step 3 says the Delta gate passed 2026-11-08. That note
      (not this file) needs updating.

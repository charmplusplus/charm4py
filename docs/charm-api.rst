
.. _charm-api-label:

charm
-----

* **charm.start(entry=None, classes=[], modules=[], interactive=False)**:

    Start the runtime system.  This is required in *all* processes, and registers
    chare types with the runtime.

    ``entry`` is the user-defined "entry point" to start the application. The runtime
    transfers control to this entry point after it has initialized. ``entry`` can be a Python
    function or a chare type.
    If it is a chare type, the runtime will create an instance of this chare on *one* PE
    (typically PE 0), and transfer control to the chare's constructor.
    The entry point function (or chare constructor) must have only one parameter, which
    is used to receive the application's arguments.

    If ``interactive`` is ``True``, an entry point is not needed and instead Charm4py
    will transfer control to a Read-Eval-Print Loop (REPL) loop on PE0.

    On calling ``charm.start()``, Charm4py automatically registers any chare types that
    are defined in the ``__main__`` module. If desired, a list of chare types can also be passed
    explictly using the ``classes`` optional parameter. These must be references to the
    classes that are to be registered, and can reference classes in other modules.
    The ``modules`` parameter can be used to automatically register any chare types defined
    in the specified list of modules. These are to be given by their names.

    .. note::
        If ``charm4py`` is imported and the program exits without calling ``charm.start()``,
        a warning will be printed. This is to remind users in case they forget
        to start the runtime (otherwise the program might hang or exit without any output).

* **charm.exit(exitCode=0)**:

    Exits the parallel program, shutting down all processes. Can be called from
    any chare or process after the runtime has started. The ``exitCode`` will
    be received by the OS on exit.

    .. note::
        Calling Python's ``exit()`` function from a chare has the same effect (Charm4py
        intercepts the SystemExit exception and calls ``charm.exit()``.

* **charm.abort(message)**:

    Aborts the program, printing the specified ``message`` and a stack
    trace of the PE which aborted. It can be called from any chare or process
    after the runtime has started.

* **charm.myPe()**:

    Returns the PE number on which the caller is currently running.

    .. note::
        Some chares can migrate between PEs during execution. As such, the value
        returned by ``myPe()`` can vary for these chares.

* **charm.numPes()**:

    Returns the total number of PEs that the application is running on.

* **charm.awaitCreation(*proxies)**:

    Makes the calling thread block until all of the
    chares in the collections referenced by the given proxies have been created on the
    system.

    .. note::
        This can only be called from within the context of a
        :ref:`threaded method <threaded-api-label>`, and the
        caller must have initiated the creation of the collections.

* **charm.createFuture(senders=1)**:

    Create and return a :ref:`Future <futures-api-label>`. The caller must be running
    within the context of a :ref:`threaded method <threaded-api-label>`.

* **charm.getTopoTreeEdges(pe, root_pe, pes=None, bfactor=4)**:

    Returns a tuple containing
    the parent PE and the list of children PEs of ``pe`` in a tree spanning the given
    ``pes``, or all PEs if ``pes`` is ``None``. If ``pes`` is specified, ``root_pe``
    must be in the first position of ``pes``, and ``pe`` must be a member of ``pes``.
    ``bfactor`` is the desired branching factor (number of children of each PE in the tree).

    In most systems, the resulting tree should be such that a physical node or host
    will have only one incoming edge (from parent).

* **charm.printStats()**:

    Print profiling metrics and statistics.
    ``charm4py.Options.PROFILING`` must have been set to ``True``.

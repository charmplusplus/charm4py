
.. _charm-api-label:

Charm
=====

The object ``charm4py.charm`` exists on every process and represents the Charm4py
runtime. It is available after importing ``charm4py``. To start the application,
you register your :ref:`Chare <chare-api-label>` classes and start the runtime
by calling ``charm.start()``.

``charm`` is also a chare,
and some of its methods can be called remotely, via its ``thisProxy`` attribute.

The API of ``charm`` is described in this section.

Start and exit
--------------

* **charm.start(entry=None, classes=[], modules=[], interactive=False)**:

    Start the runtime system.  This is required on all processes, and registers
    :ref:`Chare <chare-api-label>` classes with the runtime.

    *entry* is the user-defined entry point to start the application. The runtime
    transfers control to this entry point after it has initialized. *entry* can be a Python
    function or a chare type.
    If it is a chare type, the runtime will create an instance of this chare *on PE 0*,
    and transfer control to the chare's constructor.
    The entry point function (or chare constructor) must have only one parameter, which
    is used to receive the application arguments.

    If *interactive* is ``True``, an entry point is not needed and instead Charm4py
    will transfer control to an interactive prompt (Read-Eval-Print Loop)
    on PE 0.

    On calling ``charm.start()``, Charm4py automatically registers any chare types that
    are defined in the ``__main__`` module. If desired, a list of chare types can also be passed
    explicitly using the *classes* optional parameter. These must be the classes that
    are to be registered, and can be classes defined in other modules.
    The *modules* parameter can be used to automatically register any chare types defined
    inside the given list of modules. These modules are to be given by their names.

    .. note::
        If ``charm4py`` is imported and the program exits without calling ``charm.start()``,
        a warning will be printed. This is to remind users in case they forget
        to start the runtime (otherwise the program might hang or exit without any output).

* **charm.exit(exit_code=0)**:

    Exits the parallel program, shutting down all processes. Can be called from
    any chare or process after the runtime has started. The *exit_code* will
    be received by the OS on exit.

    *This method can be called remotely*. For this reason, it can also
    be used a callback (of reductions, etc). For example:
    ``self.reduce(charm.thisProxy.exit)``.

    .. note::
        Calling Python's ``exit()`` function from a chare has the same effect (Charm4py
        intercepts the SystemExit exception and calls ``charm.exit()``).

* **charm.abort(message)**:

    Aborts the program, printing the specified *message* and a stack
    trace of the PE which aborted. It can be called from any chare or process
    after the runtime has started.

Broadcasting globals
--------------------

* **charm.updateGlobals(globals_dict, module_name='__main__')**:

    Update the globals dictionary of module *module_name* with the key/value
    pairs from *globals_dict*, overwriting existing keys.

    *This can only be called as a remote method*.

    Example:

    .. code-block:: python

      # broadcast global 'X' to all processes, wait for completion
      charm.thisProxy.updateGlobals({'X': 333}, awaitable=True).get()

Query processor and host information
------------------------------------

* **charm.myPe()**:

    Returns the PE number on which the caller is currently running.

    .. note::
        Some chares can migrate between PEs during execution. As such, the value
        returned by ``myPe()`` can vary for these chares.

* **charm.numPes()**:

    Returns the total number of PEs that the application is running on.

* **charm.myHost()**:

    Returns the host number on which the caller is running.

* **charm.numHosts()**:

    Returns the total number of hosts on which the application is running on.

* **charm.getHostPes(host)**:

    Return the list of PEs on the specified *host* (given by host number).

* **charm.getHostFirstPe(host)**:

    Return the first PE on the specified *host* (given by host number).

* **charm.getHostNumPes(host)**:

    Return the number of PEs on the specified *host* (given by host number).

* **charm.getPeHost(pe)**:

    Return the host number on which *pe* resides.

* **charm.getPeHostRank(pe)**:

    Returns the local rank number of *pe* on the host on which it resides.

Waiting for events and completion
---------------------------------

You can obtain :ref:`Futures <futures-api-label>` when calling remote methods, to wait for
completion (see :ref:`Proxies <proxy-api-label>`).

``charm`` has the following methods related to waiting for events:

* **charm.awaitCreation(*proxies)**:

    Suspends the current coroutine until all of the chares in the collections
    referenced by the given proxies have been created on the system (in other
    words, until their constructors have been called).

    .. note::
        The coroutine must have triggered the creation of the collections.

* **charm.wait(awaitables)**:

    Suspends the current coroutine until the objects in *awaitables* become ready.
    The objects supported are :ref:`Futures <futures-api-label>` and :doc:`channels`.

* **charm.iwait(awaitables)**:

    Iteratively yield objects from *awaitables* as they become ready. The objects supported
    are :ref:`Futures <futures-api-label>` and :doc:`channels`. This can only be
    called from coroutines.

    .. warning::
        Do not suspend the coroutine until ``iwait`` has finished yielding
        all the objects.

* **charm.startQD(callback)**

    Start Quiescence Detection (QD). Quiescence is defined as the state in which
    no PE is executing a remote method, no messages are awaiting
    processing, and there are no messages in flight.
    When QD is reached, the runtime will call the *callback*. The callback
    must be a :ref:`Future <futures-api-label>` or the remote method of a chare(s)
    (specified by ``proxy.method``, where proxy can be any type of proxy,
    including a proxy to a single element or a whole collection).

* **charm.waitQD()**

    Suspend the current coroutine until Quiescence Detection is reached.

Timer-based scheduling
----------------------

* **charm.sleep(secs)**

    If this is called from a coroutine, it suspends the coroutine until at least *secs*
    seconds have elapsed (the process is free to do other work in that
    time). If it is not called from a coroutine, it is equivalent to doing
    ``time.sleep(secs)`` which puts the process to sleep.

* **charm.scheduleCallableAfter(callable_obj, secs, args=[])**

    Schedule *callable_obj* to be called after *secs* seconds. The callable can be any Python
    callable, as well as :ref:`Futures <futures-api-label>` and the remote method
    of a chare(s) (specified by ``proxy.method``, where proxy can be any type of
    proxy, including a proxy to a single element or a whole collection).
    A list of arguments can be passed via *args* (the callable will be called
    with these arguments). Note that this method only guarantees that the callable
    is called after *secs* seconds, but the exact time depends on the work
    the PE is doing.

Sections
--------

* **charm.split(proxy, numsections, section_func=None, elems=None)**:

    Split the collection referred to by *proxy* into sections. See
    :doc:`sections` for more information.

* **charm.combine(*proxies)**:

    Combine the collections referenced by *proxies* into one collection,
    returning a section proxy. See :doc:`sections` for more information.

Remote code execution
---------------------

.. note::
    These are disabled by default. Set ``charm.options.remote_exec`` to ``True``
    to enable.

* **charm.exec(code, module_name='__main__')**:

    Calls Python's ``exec(code)`` on this PE using the specified module as *globals*.
    *code* is a string containing Python code.

    *This can only be called as a remote method*.

* **charm.eval(expression, module_name='__main__')**:

    Calls Python's ``eval(expression)`` on this PE using the specified module as *globals*.

    *This can only be called as a remote method*.

Profiling
---------

* **charm.printStats()**:

    Print profiling metrics and statistics. Profiling must have been enabled
    by setting ``charm.options.profiling`` to ``True`` before calling
    ``charm.start()``. See :doc:`profiling` for more information.

    *This can be called as a remote method*.

    Example:

    .. code-block:: python

      # print stats of PE 2 and wait for completion
      charm.thisProxy[2].printStats(awaitable=True).get()


charm.options
-------------

You can set runtime options via the ``charm.options`` object, which has the
following attributes:

* **local_msg_optim** (default=True): if ``True``, remote method arguments sent to a chare
  that is in the same PE as the caller will be passed by reference (instead of copied
  or serialized).
  Best performance is obtained when this is enabled.

* **local_msg_buf_size** (default=50): size of the pool used to store "local" messages
  (see previous option).

* **pickle_protocol** (default=-1): determines the pickle protocol used by Charm4py.
  A value of ``-1`` tells ``pickle`` to use the highest protocol number (recommended).
  Note that not every type of argument sent to a remote method is pickled (see :doc:`serialization`).

* **profiling** (default=False): if ``True``, Charm4py will profile the program and
  collect timing and message statistics. See :doc:`profiling` for more information.
  Note that this will affect performance of the application.

* **quiet** (default=False): suppresses the initial Charm++ and Charm4py output.

* **remote_exec** (default=False): if ``True``, allows remote calling of ``charm.exec()``
  and ``charm.eval()``.

.. * **auto_flush_wait_queues** (default=True): if ``True``, messages or threads waiting
..   on a condition (see "when" and "wait" constructs in :ref:`chare-api-label` API) are checked and
..   flushed automatically when the conditions are met.
..   Otherwise, the application must explicitly call ``self.__flush_wait_queues__()``
..   of the chare.

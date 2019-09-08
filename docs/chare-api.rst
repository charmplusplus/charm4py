
Chares and Proxies
==================

.. _chare-api-label:

Chare
-----

An application defines new chare types by subclassing from ``charm4py.Chare``.
These can have their own custom attributes and methods.
In addition, chare classes have to be registered with the runtime when calling
``charm.start()`` (see :doc:`charm-api`).

Every chare instance has the following properties:

Attributes
~~~~~~~~~~

* **thisProxy**:

    If the chare is part of a collection, this is a proxy
    to the collection to which the element belongs to. Otherwise it is a proxy to the
    individual chare.

* **thisIndex**:

    Index of the chare in the collection to which it belongs to.

..  If the chare is not part of a collection, this attribute does not exist.


Methods
~~~~~~~

* **reduce(self, callback, data=None, reducer=None, section=None)**:

    Perform a reduction operation by giving this chare's contribution (see :doc:`reductions-api`).
    If *section* is ``None``, the reduction is performed across the elements
    of the primary collection to which the chare belongs to. If *section* is a
    section proxy, the reduction is performed across the elements of the
    section (note that the chare must belong to that section).

    The method must be called by all members to perform a successful reduction. *callback*
    will receive the result of the reduction. It can be a :ref:`Future <futures-api-label>`
    or the remote method of a chare(s) (specified by ``proxy.method``, where proxy
    can be any type of proxy, including a proxy to a single element or a whole collection).
    *data* is this chare's contribution to the reduction; *reducer* is the reducer function to apply
    (see :ref:`reducer-api-label`).

    It is possible to do "empty" reductions (if no data and reducer are given).

* **allreduce(self, data=None, reducer=None, section=None)**:

    Same as ``reduce`` but the call will return a :ref:`Future <futures-api-label>`
    which the caller can use to wait for the result (this means that the result
    of the reduction is sent to all callers). Can only be called from coroutines.

* **AtSync(self)**:

    Notify the runtime that this chare is ready for load balancing.
    If load balancing is enabled, load balancing starts on this
    PE once all of the chares that use "AtSync" have called this. When you create
    a chare array you specify if its chares use AtSync or not (see :ref:`Array <array-api-label>`).
    Load balancing starts globally once all of the PEs have started load balancing.

* **migrate(self, toPe)**:

    Requests migration of the chare to the specified PE. The chare must be
    *migratable*.

    .. caution::
        This should be called via a proxy so that it goes through the
        scheduler, for example: ``proxy.migrate(toPe)``.

        Also note that it is unusual for applications to have to manually migrate
        chares. Instead, applications should delegate to the runtime's load
        balancing framework.

* **migrated(self)**:

    This is called after a chare has migrated to a new PE. This method is empty,
    and applications can redefine it in subclasses.

* **setMigratable(self, migratable)**:

    Set whether the chare is migratable or not (*migratable* is a bool).
    If a chare is not migratable and load balancing is enabled, the load
    balancing framework will not migrate it.
    All array chares are migratable by default.


Remote methods
~~~~~~~~~~~~~~

Any user-defined methods of chares can be invoked remotely (via :ref:`proxy-api-label`).
Note that methods can also be called locally (using standard Python object method
invocation). For example, a chare might invoke one of its
own methods by doing ``self.method(*args)`` thus bypassing remote method invocation.
Note that in this case the method will be called directly and will not go through the
runtime or scheduler.


Creating single chares
----------------------

Typically, chares are created as parts of collections (see :ref:`Groups <group-api-label>`
and :ref:`Arrays <array-api-label>`).
You can, however, also create individual chares using the following syntax:

* **Chare(chare_type, args=[], onPE=-1)**:

    where *chare_type* is the type of chare you want to create.
    *args* is the list of arguments to pass to its constructor.
    If *onPE* is `-1`, the runtime decides on which PE to create it.
    Otherwise it will create the chare on the specified PE.
    This call returns a proxy.

    You can create any number of chares (of the same or different types).

    .. note::
        This call is asynchronous: it returns immediately without waiting for the
        chare to be created. See ``charm.awaitCreation()`` for one mechanism to wait
        for creation.


.. _proxy-api-label:

Proxies
-------

Proxy classes do not exist a priori. They are generated at runtime using metaprogramming,
based on the definition of the chare types that are registered when the runtime is started.

Proxy objects are returned when creating chares or collections, and are also stored
in the ``thisProxy`` attribute of chares.

.. tip::
    A proxy object is lightweight and can be sent to any chare(s) in the system via remote methods.

    Their methods can also be sent to other chares to use as callbacks (see example below).

Proxies have the same methods as the chare that they reference.
Calling those methods will result in the method being invoked on the chare(s) that
the proxy references, regardless of the location of the chare.

The syntax to call a remote method is:

**proxy.remoteMethod(*args, **kwargs, awaitable=False, ret=False)**:

    Calls the method of the chare(s) referenced
    by the proxy. This is a remote method invocation. If the proxy references a
    collection, a broadcast call is made and the method is invoked on all chares
    in the collection. Otherwise, the method is called on an individual chare.
    The call returns immediately and does not wait for the method to be invoked at the
    remote chare(s).

    If *awaitable* is ``True``, the call returns a :ref:`Future <futures-api-label>`,
    which can be used to wait for completion. This also works for broadcast
    calls (wait for the call to complete on every element).

    If *ret* is ``True``, the call returns a :ref:`Future <futures-api-label>`,
    which can be used to wait for the result. This also works for broadcast calls. In this
    case, the return value will be a list of return values, sorted by element index.

    If *ret* or *awaitable* are ``True`` and the remote method throws an
    unhandled exception, the exception is propagated to the caller (even if the
    caller is in another PE). The exception is raised at the caller when
    it queries the future.

Proxies the refer to collections can be **sliced** to obtain section proxies
(see :doc:`sections`).

All proxies implement ``__eq__`` and ``__hash__``, with correct results
between proxies generated locally and those obtained from a remote PE.
This allows, for example, checking proxies for equality, using them as
dictionary keys or inserting in sets.


Example
~~~~~~~

.. code-block:: python

    from charm4py import charm, Chare, Group

    class A(Chare):

        def start(self):
            b_proxy = Chare(B)
            # call work and send one of my methods to use as callback
            b_proxy.work(self.thisProxy.recvResult)

        def recvResult(self, result):
            print('Result is', result)
            exit()

    class B(Chare):

        def work(self, callback):
            # ... do work ...
            result = ...
            callback(result)

    def main(args):
        a_proxy = Chare(A)
        a_proxy.start()

    charm.start(main)

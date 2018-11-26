
.. _chare-api-label:

Chare
-----

An application specifies new chare types by defining classes that
inherit from ``charm4py.Chare``. These classes can have custom attributes
and methods. In addition, every chare instance has the following properties:

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

* **contribute(self, data, reducer, target)**:

    Contribute data for a reduction operation
    across the elements of the collection to which the chare belongs to. The method
    must be called by all members to perform a successful reduction. ``data`` is the data to reduce;
    ``reducer`` is the reducer function to apply (see :ref:`reducer-api-label`); ``target`` will receive
    the result of the reduction. The target can be the remote method of any chare(s)
    (indicated by ``proxy.method``) or a :ref:`Future <futures-api-label>`.

* **wait(self, condition)**:

    Pauses the current thread until the specified
    condition is true. ``condition`` is a string containing a Python conditional statement.
    The conditional statement can reference attributes of the chare, constants and globals,
    but not local names in the caller's frame. To use this construct, the caller must be
    running within the context of a threaded method (see below).

* **migrate(self, toPe)**:

    Requests migration of the chare to the specified PE. Note that this should be
    the last instruction executed by the chare's current call stack. The chare must be
    *migratable*.

Remote methods (aka entry methods)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any user-defined methods of the chare type can be invoked remotely (via a :ref:`proxy-api-label`).
Note that methods can also be called locally (using standard Python object method
invocation). For example, a chare might invoke one of its
own methods by doing ``self.method(*args)`` thus bypassing remote method invocation.
Note that in this case the method will be called directly and will not go through the
runtime or scheduler.

.. _threaded-api-label:

"threaded" method decorator
+++++++++++++++++++++++++++

Methods tagged with this decorator will run in their own thread when called
via a proxy. This allows pausing the execution of the method to wait for certain events
(see ``wait`` construct above, :ref:`futures-api-label` or ``charm.awaitCreation()``).

The decorator is placed before the definition of the method, using the syntax:
``@charm4py.threaded``

.. note::
    While a thread is paused, the runtime continues scheduling other work in the same
    process, even for the same chare.

.. important::
    The application entry point is always threaded.

"when" method decorator
+++++++++++++++++++++++

The semantics of when a remote method can be invoked *at the receiver* can be
controlled using the ``when`` decorator. The decorator is placed before the definition
of the method, using the syntax:

``@charm4py.when('condition')``

where ``condition`` is a string containing a standard Python conditional statement. The statement
can reference any of the chare's attributes (prefixed by ``self``), as well as any of the
method's arguments (referenced by their name).

.. important::
    Callers are free to invoke the method whenever they are ready, without having
    to wait for the receiver (callee) to be ready. The
    message will be delivered and buffered at the receiving side until the receiver
    is ready and the condition is met. This is desirable for performance reasons.


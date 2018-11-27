
.. _futures-api-label:

Futures
-------

Futures are objects that act as placeholders for values which are unknown at the time
they are created. A future is an instance of ``charm4py.threads.Future``.

Creation
~~~~~~~~

Futures can be returned when invoking remote methods (see :ref:`proxy-api-label`).
This allows the caller to continue doing work and wait for the value at the caller's
convenience.

Futures can also be created explicitly by calling ``charm.createFuture(senders=1)``,
which returns a new future object accepting
``senders`` number of values. A future created in this
way can be sent to any chare(s) in the system by message passing, with the purpose
of allowing remote chares to send values to the caller.

.. note::
    Futures can only be created from threads other than the main thread (see `threaded entry methods`__).

.. __: chare-api.html#threaded-method-decorator


"Future" methods
~~~~~~~~~~~~~~~~

* **get(self)**:

    Return the value of the future, or list of values if created with
    ``senders > 1``. The call will block if the value(s) has not yet been received.
    This can only be used by the chare that created the future.

* **send(self, value)**:

    Send ``value`` to the chare waiting on the future. Can be called
    from any chare.

Future as reduction target
~~~~~~~~~~~~~~~~~~~~~~~~~~

A future can be used as a reduction target (see :ref:`chare-api-label`). In this case,
the result of the reduction will be sent to the future.

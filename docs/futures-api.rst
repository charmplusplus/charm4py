
.. _futures-api-label:

Futures
-------

Futures are objects that act as placeholders for values which are unknown at the time
they are created. Their main use is to allow a coroutine to suspend, waiting
until a message or value becomes available without having to exit the coroutine,
and without blocking the rest of the coroutines/tasks in its process.


Creation
~~~~~~~~

Futures can be returned by the runtime when invoking remote methods (see :ref:`proxy-api-label`).
This allows the caller to continue doing work and wait for the return value at
the caller's convenience, or wait for the method to complete.

Futures can also be created explicitly by calling:

* **charm4py.Future(num_vals=1)**:

    Returns a new future object accepting *num_vals* number of values. A future
    created in this way can be sent to any chare(s) in the system by message
    passing, with the purpose of allowing remote chares to send values to its origin.

    .. note::
        Futures can only be created from coroutines.


Methods
~~~~~~~

* **get(self)**:

    Return the value of the future, or list of values if created with
    ``senders > 1``. The call will block if the value(s) has not yet been received.
    This can only be called from a coroutine, by the chare that created the future.

    *If a future receives an Exception, it will raise it on calling this method.*

* **send(self, value=None)**:

    Send *value* to the chare waiting on the future. Can be called
    from any chare.

* **__call__(self, value=None)**:

    This makes futures **callable**, providing a generic callback interface.
    Calling a future is the same as using the ``send()`` method.

Future as callback
~~~~~~~~~~~~~~~~~~

Futures are callable (see above) and can also be used as reduction callbacks.


Example
~~~~~~~

.. code-block:: python

    from charm4py import charm, Chare, Array, Future, Reducer

    class A(Chare):

        def work(self, future):
            # ...
            result = # ...
            # use future as reduction callback (send reduction result to future)
            self.reduce(future, result, Reducer.sum)

    def main(args):
        array_proxy = Array(A, charm.numPes() * 8)
        f = Future()
        array_proxy.work(f)
        result = f.get()  # wait for work to complete on all chares in array
        exit()

    charm.start(main)

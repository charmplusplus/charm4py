============
Introduction
============

Charm4py's programming model is based on an actor model.
Distributed objects in Charm4py are called *Chares* (pronounced char).
A chare is essentially a Python object in the OOP sense with its own attributes (data) and methods.
A chare lives on one process, and some chares can migrate between processes (e.g.
for dynamic load balancing).
Chares can call methods of any other chares in the system via remote method invocation,
with the same syntax as calling regular Python methods.
The runtime automatically takes care of location management and uses the most
efficient technique for method invocation and message passing.
Parallelism is achieved by having chares distributed across processes/cores.

You can create as many collections of distributed objects as you want, of the same
or different types. There can be multiple chares on one process, each executing one or multiple tasks.
Having many chares per core can help the runtime maximize resource utilization,
dynamically balance load and overlap communication and computation.

In addition, Charm4py supports the following features to facilitate expression of concurrency:
coroutines, channels and futures. These are seamlessly integrated into the actor model.

We will show some simple examples now to quickly illustrate these concepts.
For a more step-by-step tutorial, you can check the :doc:`tutorial` which can be done in an
interactive session.


Actor model
-----------

Chares are defined as regular Python classes that are a subclass of ``Chare``:

.. code-block:: python

    from charm4py import charm, Chare, Array

    # Define my own chare type A (instances will be distributed objects)
    class A(Chare):

        def start(self):
            # call method 'sayHi' of element 1 in my Array
            self.thisProxy[1].sayHi('hello world')

        def sayHi(self, message):
            print(message, 'on process', charm.myPe())
            exit()

    def main(args):
        # create a distributed Array of 2 objects of type A
        array_proxy = Array(A, 2)
        # call method 'start' of element 0 of the Array
        array_proxy[0].start()

    # start the Charm runtime. after initialization, the runtime will call
    # function 'main' on the first process
    charm.start(main)


One important thing to note here is that in Charm4py *every remote method invocation is asynchronous*.
This allows the runtime to maximize resource efficiency and
overlap communication and computation. This also means that calls will return immediately.
You can, however,
request a future when calling remote methods, and use the future to suspend
the current coroutine until the remote method completes, or to obtain a return value (more on this
below).

Coroutines
----------

Chare methods can act as coroutines, which simply means that they can
suspend their execution to wait for events/messages, and continue where
they left off when the event arrives. This can allow writing significant
parts of your code in direct or sequential style. Simply decorate a method
with ``@coro`` to allow it to work as a coroutine.
When a coroutine suspends, the runtime is free to schedule other work on the same process,
even for the same chare.

.. and other instances of the same coroutine.

Coroutines are typically used in conjunction with channels and futures (described below).


Channels
--------

Channels establish streamed connections between chares (currently one-to-one).
Messages can be sent/received to/from the channel using the methods ``send()``
and ``recv()``. The following example uses Channels and coroutines:

.. code-block:: python

    from charm4py import charm, Chare, Array, coro, Channel

    class A(Chare):

        @coro
        def start(self):
            if self.thisIndex == (0,):
                # I am element 0, establish a Channel with element 1 of my Array
                ch = Channel(self, remote=self.thisProxy[1])
                # send msg on the channel (this is asynchronous)
                ch.send('hello world')
            else:
                # I am element 1, establish a Channel with element 0 of my Array
                ch = Channel(self, remote=self.thisProxy[0])
                # receive msg from the channel. coroutine suspends until the msg arrives
                print(ch.recv())
                exit()

    def main(args):
        a = Array(A, 2)
        # call method 'start' of every element of the array (this is a broadcast)
        a.start()

    charm.start(main)


.. tip::
  Coroutine methods are currently implemented using greenlets, which are very lightweight.
  The amount of overhead they add is tiny, so don't hesitate to use them
  where appropiate. Also note that the runtime will tell you if ``@coro`` is needed.

.. Nonetheless, it was decided not to make every method a coroutine
.. by default because some objects can have very small methods (computationally speaking).

Futures
-------

Coroutines can also create futures and use them to wait for certain
events/messages. A future can be sent to other chares in the system, and any
chare can send a value to the future, which will resume the coroutine that
was waiting on it. For example:

.. code-block:: python

    from charm4py import charm, Chare, Array, coro, Channel, Future

    class A(Chare):

        @coro
        def start(self, done):
            neighbor = self.thisProxy[(self.thisIndex[0] + 1) % 2]
            # establish a channel with my neighbor
            ch = Channel(self, remote=neighbor)
            # each chare sends and receives a msg to/from its neighbor for 10 steps
            for i in range(10):
                ch.send(i)
                assert ch.recv() == i
            if self.thisIndex == (0,):
                # signal the future that we are done
                done()

    def main(args):
        a = Array(A, 2)
        # create a Future
        done = Future()
        # call start method on both elements (broadcast), passing the future
        a.start(done)
        # ... do work ...
        # 'get' suspends the coroutine until the future receives a value
        # (note that the main function is always a coroutine)
        done.get()
        exit()

    charm.start(main)


Awaitable remote method calls
-----------------------------

As mentioned above, you can also obtain a future when invoking a remote method of
any chare. This is done by using the keywords ``awaitable=True`` and
``ret=True`` when calling the method.
The former specifies that the call is awaitable and allows waiting for completion.
The latter specifies that the caller wants to receive the return value(s).
Note that ``ret=True`` automatically implies that the call is awaitable (a return
value can only be received after the call has completed).

Example:

.. code-block:: python

    from charm4py import charm, Chare, Array

    class A(Chare):

        def work(self):
            result = # ... do some work ...
            return result

    def main(args):
        a = Array(A, 2)
        future = a[1].work(ret=True)
        # ... can do other stuff while the remote chare works ...
        # query future now. will suspend 'main' if the value has not arrived yet
        value = future.get()
        print('Result is', value)
        exit()

    charm.start(main)


.. caution::
  For broadcasts, ``ret=True`` will cause a list of return values to be sent to the caller.
  This is more expensive than simply waiting for completion
  of the broadcast with ``awaitable=True``, and can also result in very long lists of return
  values if you are broadcasting to thousands of chares. In summary,
  only use ``ret=True`` for broadcasts if a list of return values is what you want.

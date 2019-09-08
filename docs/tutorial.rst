========
Tutorial
========

.. contents::

This is a step-by-step tutorial to introduce the main concepts of Charm4py, and
is meant to be done from an interactive session.
It is not meant to provide realistic examples, or to cover every possible topic.
For examples, you can refer to :doc:`examples`.

.. This tutorial assumes that you have installed Charm4py as described in :doc:`install`.

To begin, launch an interactive session with 2 processes::

    $ python3 -m charmrun.start +p2 ++interactive


This launches Charm4py with two processes on the local host, with an interactive
console running on the first process. In Charm4py, we also refer to processes
as Processing Elements (PEs).

First steps
-----------

The interactive console is actually a chare running on PE 0, and the prompt
is running inside a coroutine of this chare. Typing::

    >>> self
    <charm4py.interactive.InteractiveConsole object at 0x7f7d9b1290f0>

will show that ``self`` is an InteractiveConsole object. As mentioned, this
object exists only on PE 0.

Now, let's look at the ``charm`` object::

    >>> charm
    <charm4py.charm.Charm object at 0x7f7d9f6d9208>

``charm`` exists on every PE. It represents the Charm runtime. We can query
information from it::

    >>> charm.myPe()
    0

Tells us that this process is PE 0.

    >>> charm.numPes()
    2
    >>> charm.numHosts()
    1

The above tells us that we are running Charm4py with 2 PEs on 1 host.


Chares
------

In this tutorial, we are going to be defining chares dynamically
after the Charm runtime has started, and so these definitions need to be sent
to other processes at runtime.
Note that non-interactive applications typically have everything defined in the
source files (which every process reads at startup).

Let's define a simple chare type. Paste the following in the console:

.. code-block:: python

    class Simple(Chare):
        def sayHi(self):
            print('Hello from PE', charm.myPe())
            return 'hi done'

You will see this::

    Charm4py> Broadcasted Chare definition

We have defined a new chare of type ``Simple`` and the runtime has automatically broadcasted
its definition to other processes. We can now create chares of this type and
call their methods::

    >>> chare = Chare(Simple, onPE=1)  # create a single chare on PE 1
    >>> chare
    <__main__.SimpleArrayProxy object at 0x7f7d9b129668>
    >>> chare.sayHi()
    Hello from PE 1

It is important to note that ``chare`` is what is called a
:ref:`Proxy <proxy-api-label>`. As we can
see, remote methods are called via proxies, using regular Python method
invocation syntax.

.. A proxy has the same methods as the chare(s) that it references.

.. tip::
    Proxies are lightweight objects that can be sent to other chares.

The chare we created lives on PE 1, and that is where
its method executes. Note that Charm4py automatically collects "prints" and
sends them to PE 0, where they are actually printed.

Remote method invocation is asynchronous, returns immediately,
and by default does not return anything. We can wait for a call to complete or
obtain a return value by requesting a :ref:`Future <futures-api-label>` using ``ret=True``::

    >>> f = chare.sayHi(ret=True)
    Hello from PE 1
    >>> f
    <charm4py.threads.Future object at 0x7f7d9b129f28>
    >>> f.get()
    'hi done'


Remote method invocation is asynchronous
----------------------------------------

All method invocations via a proxy are *asynchronous*. Above, we called some
remote methods, but they execute so quickly that it is not obvious that
it happens asynchronously. To illustrate this more clearly, we will define a
method that takes longer to execute.

Paste the following into the console:

.. code-block:: python

    class AsyncSimple(Chare):
        def sayHi(self):
            time.sleep(5)
            print('Hello from PE', charm.myPe())
            return 'hi done'

Now, let's invoke the method::

    >>> import time
    Charm4py> Broadcasted import statement
    >>> chare = Chare(AsyncSimple, onPE=1)
    >>> chare.sayHi()

As we can see, the call returns immediately. We won't see any output until
the method completes (after 5 seconds).
Now let's see what happens if we want to explicitly wait for the call to complete::

    >>> f = chare.sayHi(awaitable=True)
    >>> f.get()

We request a future by making the call ``awaitable``. We can then block on the future
to wait for completion. **It is important to note that this only blocks the
current coroutine** (it does not block the whole process).

Charm also has a nice feature called *quiescence detection* (QD) that can be used to detect
when all PEs are idle. We can wait for QD like this::

    >>> chare.sayHi()
    >>> charm.waitQD()


Chare Groups
------------

In many situations we create *collections* of chares, which are distributed across
processes by the runtime.
First let's look at **Groups**, which are collections with one element per PE::

    >>> g = Group(AsyncSimple)
    >>> g
    <__main__.AsyncSimpleGroupProxy object at 0x7f7d9f9f7fd0>
    >>> g.sayHi(awaitable=True).get()
    Hello from PE 0
    Hello from PE 1

We created a group of AsyncSimple chares and made an awaitable call. Note that
because we don't refer to any specific element, the message is sent to every
member (also known as a *broadcast*). We call ``get()`` on the obtained future,
which blocks until the call completes on every member of the group. Note that
we didn't get any return values. Let's request return values now::

    >>> g.sayHi(ret=True).get()
    Hello from PE 1
    Hello from PE 0
    ['hi done', 'hi done']

As we can see, we got return values from every member. We can refer to specific
members by using their index on the proxy. For groups, the index coincides with the
PE number::

    >>> g[1].sayHi(ret=True).get()
    'hi done'
    Hello from PE 1

Chares have one primary collection to which they can belong to, and they have
access to the collection proxy via their ``thisProxy`` attribute. They
have access to their index in the collection via the ``thisIndex`` attribute.
For example, define the following chare type:

.. code-block:: python

    class Test(Chare):
        def start(self):
            print('I am element', self.thisIndex, 'on PE', charm.myPe(),
                  'sending a msg to element 1')
            self.thisProxy[1].sayHi()
        def sayHi(self):
            print('Hello from element', self.thisIndex, 'on PE', charm.myPe())

Now, we will make element 0 send a message to element 1::

    >>> g = Group(Test)
    >>> g[0].start()
    I am element 0 on PE 0 sending a msg to element 1
    Hello from element 1 on PE 1

You can store a proxy referencing an individual element, for later use::

    >>> elem = g[0]
    >>> elem.sayHi()
    Hello from element 0 on PE 0

Chare Arrays
------------

Chare Arrays are a more versatile kind of distributed collection, which can have
zero or multiple chares on a PE, and chares can migrate between processes.

Let's create an Array of 4 chares of the previously defined type ``Test`` and
see where the runtime places them::

    >>> a = Array(Test, 4)
    >>> a.sayHi()
    Hello from element (2,) on PE 1
    Hello from element (3,) on PE 1
    Hello from element (0,) on PE 0
    Hello from element (1,) on PE 0

As we can see, it has created two on each PE.

Array elements have N-dimensional indexes (from 1D to 6D), represented by
a tuple. For example, let's create a 2 x 2 array instead::

    >>> a = Array(Test, (2,2))
    >>> a.sayHi()
    Hello from element (0, 0) on PE 0
    Hello from element (0, 1) on PE 0
    Hello from element (1, 0) on PE 1
    Hello from element (1, 1) on PE 1
    >>> a[(1,0)].sayHi()
    Hello from element (1, 0) on PE 1


Charm is a chare too
--------------------

The ``charm`` object is a chare too (part of a Group), which means it has methods that can
be invoked remotely::

    >>> charm.thisProxy[1].myPe(ret=True).get()
    1

Calls the method ``myPe()`` of ``charm`` on PE 1, and returns the value.

In interactive mode, Charm also exposes ``exec`` and ``eval`` for dynamic
remote code execution::

    >>> charm.thisProxy[1].eval('charm.myPe()', ret=True).get()
    1

Note that remote exec and eval are only enabled by default in interactive mode.
If you want to use them in regular non-interactive mode, you have to set
``charm.options.remote_exec`` to ``True`` before the charm runtime is started.

Broadcasting globals
--------------------

Suppose we want to broadcast and set globals on some or all processes. With what we
know, we could easily implement our own way of doing this. For example, we
could create a custom chare Group with a method that receives objects and
stores them in the global namespace. However, charm provides a convenient
remote method to do this::

    >>> charm.thisProxy.updateGlobals({'MY_GLOBAL': 1234}, awaitable=True).get()
    >>> charm.thisProxy.eval('MY_GLOBAL', ret=True).get()
    [1234, 1234]

As we can see, there is now a global called ``MY_GLOBAL`` in the main module's
namespace on every PE. We can specify the Python module where we want to set
the global variables as a second parameter to ``updateGlobals``. If left unspecified,
it will use ``__main__`` (which is the same namespace where InteractiveConsole
runs).

Reductions
----------

Reductions are very useful to aggregate data among members of a collection in
a way that is scalable and efficient, and send the results anywhere in
the system via a callback.
We will illustrate this with a simple example. First define the following chare type:

.. code-block:: python

    class RedTest(Chare):
        def work(self, data, callback):
            self.reduce(callback, data, Reducer.sum)
        def printResult(self, result):
            print('[' + str(self.thisIndex[0]) + '] Result is', result)

Now we will create an Array of 20 of these chares and broadcast some data so that
they can perform a "sum" reduction.
Normally, each chare would provide its own unique data to a reduction, but in this
case we broadcast the value for simplicity.
As callback, we will provide a future::

    >>> a = Array(RedTest, 20)
    >>> f = Future()
    >>> a.work(1, f)
    >>> f.get()
    20

We manually created a future to receive the result, and passed data (int value 1) and the future via a
broadcast call. The chares performed a reduction using the received data, and sent
the result to the callback, in this case the future. Because we passed a value
of 1, the result equals the number of chares. Note that **reductions happen asynchronously**,
and don't block other ongoing tasks in the system.


.. note::
  Reductions are performed in the context of the collection to which the chare belongs
  to: all objects in that particular collection have to contribute for
  the reduction to complete.

The other main type of callback used in Charm is a remote method of some chare(s).
For example, we can send the result of the reduction to element 7 of the array::

    >>> a.work(1, a[7].printResult)
    [7] Result is 20

You can even broadcast the result of the reduction to all elements using ``a.printResult`` as
the callback. Try it and see what happens.

Reductions are useful when data that is distributed among many objects across the
system needs to be aggregated in some way, for example to obtain the maximum value
in a distributed data set or to concatenate data in some fashion. The aggregation
operations that are applied to the data are called **reducers**, and Charm4py includes
several built-in reducers, including sum, max, min, product and gather. Users can
also define their own reducers (see :ref:`Reducers <reducer-api-label>`).

It is common to perform reduction operations on arrays::

    >>> import numpy
    >>> f = Future()
    >>> a.work(numpy.array([1,2,3]), f)
    >>> f.get()
    array([20, 40, 60])


You can also do *empty reductions* to know when all the elements in a collection
have reached a certain point. Simply provide a callback to the ``reduce`` call
and omit the data and reducer.


Channels
--------

Channels in Charm4py are streams or pipes between chares (currently only
point-to-point). They are useful for writing iterative applications where chares
always send/recv to/from the same the set of chares.

Here, we will establish a channel between the InteractiveConsole and another
chare. First let's define the chare:

.. code-block:: python

    class Echo(Chare):
        @coro
        def run(self, remote_proxy):
            ch = Channel(self, remote=remote_proxy)
            while True:
                x = ch.recv()
                ch.send(x)

Echo chares will establish a channel with whatever chare is passed to
them in the ``run`` method, and will enter an infinite loop where they
wait to receive something from the channel and then send it right back::

    >>> chare = Chare(Echo, onPE=1)
    >>> chare.run(self.thisProxy)
    >>> ch = Channel(self, remote=chare)
    >>> ch.send('hello')
    >>> ch.recv()
    'hello'
    >>> ch.send(1,2,3)
    >>> ch.recv()
    (1, 2, 3)

Note that on calling ``recv()`` a coroutine suspends until there is something
to receive.


Pool
----

Charm4py also has a distributed pool of workers that can be used to execute transient
tasks in parallel, where tasks are defined as Python functions. This pool
automatically distributes tasks across processes and even multiple hosts.

A common operation is ``map``, which applies a function in parallel to the
elements of an iterable and returns the list of results. For example::

    >>> charm.pool.map(abs, range(-1,-20,-1))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

If your tasks are very small, you will want to group them into chunks for
efficiency. Pool can do this for you with the ``chunksize`` parameter
(see :doc:`pool`).

Note that the pool of workers reserves PE 0 for a scheduler, so there are
P-1 workers (P being the number of PEs). So you might want to adjust the
number of processes accordingly.

.. tip::
  Tasks themselves can use the pool to create and wait for other tasks, which
  is useful for implementing recursive parallel algorithms and state space
  search (or similar) algorithms. There are examples of this in the source
  code repository.

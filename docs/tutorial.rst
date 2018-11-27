========
Tutorial
========

.. contents::

This tutorial assumes that you have installed Charm4py as described in :doc:`install`.
You can run any of these examples in an interactive Python shell (using multiple processes)
by launching Charm4py in the following manner::

    $ python3 -m charmrun.start +p4 ++interactive

and inserting code at the prompt. Note that in interactive mode the runtime is already
started when the interactive shell appears, so ``charm.start()`` does *not* need to be called.
For the examples below, you can directly call the main function or, alternatively, just run the body of the main
function in the top-level shell.


Program start and exit
----------------------

To start a Charm program, you need to invoke the ``charm.start(entry)`` method.
We will begin with a simple example:

.. code-block:: python

    # start.py
    from charm4py import charm

    def main(args):
        print("Charm program started on processor", charm.myPe())
        print("Running on", charm.numPes(), "processors")
        exit()

    charm.start(main)  # call main([]) in interactive mode


We need to define an entry point to the Charm4py program, which we refer to as the
Charm *main* function.
In our example, it is the function called ``main`` .
The main function runs on only one processor, typically processor 0, and is in charge
of starting the creation and distribution of work across the system. The main function must take
one argument to get the list of command-line arguments.
In this example, we are specifying the
function ``main`` as the main function by passing it to the ``start`` method.

The method ``numPes`` returns the number of processors (aka Processing Elements) on
which the distributed program is running. The method ``myPe`` returns the processor
number on which the caller resides.

An explicit call to ``exit()`` is necessary to finish the parallel program, shutting down all
processes. It can be called from any chare on any processor.

To launch the example with charmrun using 4 processes::

    $ python -m charmrun.start +p4 start.py


Defining Chares
---------------

Chares are distributed objects that make up the parallel application (see :doc:`overview`).
To define a Chare, simply define a class that is a subclass of ``Chare``.

.. code-block:: python

    from charm4py import Chare

    class MyChare(Chare):

        def __init__(self):
            # chare initialization code here

        def work(self, data):
            # ... do something ...

Any methods of ``MyChare`` will be remotely callable by other chares.

For easy management of distributed objects, you can organize chares into distributed collections:


.. code-block:: python

    # chares.py
    from charm4py import charm, Chare, Group, Array

    class MyChare(Chare):
        def __init__(self):
            print("Hello from MyChare instance in processor", charm.myPe())

        def work(self, data):
          pass

    def main(args):

        # create one instance of MyChare on every processor
        my_group = Group(MyChare)

        # create 3 instances of MyChare, distributed among the cores by the runtime
        my_array = Array(MyChare, 3)

        # create 2 x 2 instances of MyChare, indexed using 2D index and distributed
        # among all cores by the runtime
        my_2d_array = Array(MyChare, (2, 2))

        charm.awaitCreation(my_group, my_array, my_2d_array)
        exit()

    charm.start(main)  # call main([]) in interactive mode

The above program will create P + 3 + 2\*2 chares and print a message for each created
chare, where P is the number of processors used to launch the program.
This is the output for 2 PEs:

.. code-block:: text

    $ python -m charmrun.start +p2 chares.py ++quiet
    Hello from MyChare instance in processor 0
    Hello from MyChare instance in processor 0
    Hello from MyChare instance in processor 0
    Hello from MyChare instance in processor 0
    Hello from MyChare instance in processor 0
    Hello from MyChare instance in processor 1
    Hello from MyChare instance in processor 1
    Hello from MyChare instance in processor 1
    Hello from MyChare instance in processor 1

It is important to note that creation of chares across the system happens asynchronously.
In other words, when the above calls to create collections return,
the chares have not yet been created on all PEs. The ``awaitCreation`` method is
used to wait for all the chares in the specified collections to be created.

.. note::
    Chares can be created at any point once the Charm *main* function has been reached.

If a program defines new Chare types in files other than the one used to launch the
application, the user needs to pass the names of those modules when starting charm.
For example:

.. code-block:: python

    charm.start(main, ['module1', 'module2'])


Remote method invocation
------------------------

To invoke methods on chares, a remote reference or *proxy* is needed. A proxy has the same
methods as the chare that it references. For example, assuming we have a proxy to a
``MyChare`` object, we can call method ``work`` like this:

.. code-block:: python

    # invoke method 'work' on the chare, passing list [1,2,3] as argument
    proxy.work([1,2,3])

Any number and type of arguments can be used, and the runtime will take care of sending
the arguments if the destination is on a different host. We will also refer to
invoking a remote method as sending a message.

.. warning::

    Make sure that the caller does not modify any objects passed as arguments
    after making the call. It also should not attempt to reuse them if the callee is
    expected to modify them.
    The caller can safely discard any references to these objects if desired.

References to collections serve as proxies to their elements. For example,
``my_group`` above is a proxy to the group and its elements. To invoke a method on
all elements in the group do:

.. code-block:: python

    my_group.work(x)    # 'work' is called on every element

To invoke a method on a particular element do:

.. code-block:: python

    my_group[3].work(x)  # call 'work' on element with index 3

To store a proxy referencing an individual element for later use:

.. code-block:: python

    elem_3_proxy = my_group[3]
    elem_3_proxy.work(x)   # call 'work' on element with index 3 in my_group

The above also applies to Chare Arrays. In the case of N-dimensional array indexes:

.. code-block:: python

    my_array[10,10].work(x)	# call 'work' on element (10,10)

.. tip::
    Proxies can be sent to other chares as arguments of methods.

For performance reasons, method invocation is always *asynchronous* in Charm4py, i.e. methods
return immediately without waiting for the actual method to be invoked on the remote
object, and therefore without returning any result. Asynchronous method invocation
is desirable because it leads to better overlap of computation and communication, and better
resource utilization (which translates to more speed). Note that this does not mean
that we cannot obtain a result from a remote chare as a result of calling
one of its methods. There are two ways of doing this:

*1. Using Futures:*

The user can request to obtain a future_ as a result of calling a remote method, by
using the keyword ``ret``:

.. _future: https://en.wikipedia.org/wiki/Futures_and_promises


.. code-block:: python

    def work(self):
        # call method 'apply' of chares with index (10,10) and (20,20), requesting futures
        future1 = my_array[10,10].apply(3, ret=True)
        future2 = my_array[20,20].apply(3, ret=True)

        # ... do more work ...

        # I need the results now, call 'get' to obtain them. Will block until they arrive,
        # or return immediately if the result has already arrived
        x = future1.get()
        y = future2.get()

        # call 'apply' and block until result arrives
        z = my_array[10,10].apply(5, ret=True).get()

    def apply(self, x):
        self.data += x          # apply parameter
        return self.data.copy() # return result to caller

The ``get`` method of a future will block the thread on the caller side while it waits for the result, but it
is important to note that it does not block the whole process. Other available work in
the process (including of the same chare that blocked) will continue to be executed.


*2. With remote method invocation:*

.. code-block:: python

    # --- in chare 0 ---
    def work(self):
        group[1].apply(3) # tell chare 1 to apply 3 to its data, returns immediately

    def storeResult(self, data):
        # got resulting data from remote object
        # do something with data

    # --- in chare 1 ---
    def apply(self, x):
      self.data += x  # apply parameter
      group[0].storeResult(self.data.copy())  # return result to caller


Reductions 101
--------------

Reductions can be performed by members of a collection with the result being sent to
any chare or future of your choice.

.. code-block:: python

    # reduction.py
    from charm4py import charm, Chare, Group, Reducer

    class MyChare(Chare):

        def work(self, data):
            self.contribute(data, Reducer.sum, self.thisProxy[0].collectResult)

        def collectResult(self, result):
            print("Result is", result)
            exit()

    def main(args):
        my_group = Group(MyChare)
        my_group.work(3)

    charm.start(main)  # call main([]) in interactive mode


In the above code, every element in the group contributes the data received from
main (int of value 3) and the result
is added internally by Charm and sent to method ``collectResult`` of the first chare in the group
(to the chare in processor 0 because Groups have one chare per PE).
Chares that are members of a collection have an attribute called ``thisProxy`` that
is a proxy to said collection.

For the above code, the result of the reduction will be 3 x number of cores.

Reductions are performed in the context of the collection to which the chare belongs
to: all objects in that particular collection have to contribute for the reduction
to finish.

.. hint::
    Reductions are highly optimized operations that are performed by the runtime in
    parallel across hosts and processes, and are designed to be scalable up to the largest
    systems, including supercomputers.

Reductions are useful when data that is distributed among many objects across the
system needs to be aggregated in some way, for example to obtain the maximum value
in a distributed data set or to concatenate data in some fashion. The aggregation
operations that are applied to the data are called *reducers*, and Charm4py includes
several built-in reducers (including ``sum``, ``max``, ``min``, ``product``, ``gather``),
as well as allowing users to easily define their own custom reducers for use in reductions.
Please refer to the manual for more information.

Arrays (array.array_) and `NumPy arrays`_ can be passed as contribution to many of
Charm4py's built-in reducers. The reducer will be applied to elements
having the same index in the array. The size of the result will thus be the same as
that of each contribution.

For example:

.. code-block:: python


    def doWork(self):
        a = numpy.array([0,1,2])  # all elements contribute the same data
        self.contribute(a, Reducer.sum, target.collectResult)

    def collectResult(self, a):
        print(a)  # output is array([0, 4, 8]) when 4 elements contribute



.. _array.array: https://docs.python.org/3/library/array.html

.. _NumPy arrays: https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html




Hello World
-----------

Now we will show a full *Hello World* example, that prints a message from all processors:

.. code-block:: python

    # hello_world.py
    from charm4py import Chare, Group, charm

    class Hello(Chare):

        def SayHi(self):
            print("Hello World from element", self.thisIndex)

    def main(args):
        # create Group of Hello objects (one object exists and runs on each core)
        hellos = Group(Hello)
        # call method 'SayHi' of all group members, wait for method to be invoked on all
        hellos.SayHi(ret=True).get()
        exit()

    charm.start(main)  # call main([]) in interactive mode



The *main* function requests the creation of a ``Group`` of chares of type ``Hello``.
As explained above, group creation is asynchronous and as
such the chares in the group have not been created yet when the call returns.
Next, *main* tells all the members of the group to say hello, and blocks until
the method is invoked on all members, because we don't want to exit the program
until this happens. This is achieved by requesting a future (using
``ret=True``), and waiting until the future resolves by calling ``get``.

When the ``SayHi`` method is invoked on the remote chares, they print their message along
with their index in the collection (which is stored in the attribute ``thisIndex``).
For groups, the index is an ``int`` and coincides with the PE number on which the chare
is located. For arrays, the index is a ``tuple``.

In this example, the runtime internally performs a reduction to know when all the group
elements have concluded and sends the result to the *future*. The same effect can be achieved
explicitly by the user like this:

.. code-block:: python

    # hello_world2.py
    from charm4py import Chare, Group, charm

    class Hello(Chare):

        def SayHi(self, future):
            print("Hello World from element", self.thisIndex)
            self.contribute(None, None, future)

    def main(args):
        # create Group of Hello objects (one object exists and runs on each core)
        hellos = Group(Hello)
        # call method 'SayHi' of all group members, wait for method to be invoked on all
        f = charm.createFuture()
        hellos.SayHi(f)
        f.get()
        exit()

    charm.start(main)  # call main([]) in interactive mode

As we can see, here the user explicitly creates a future and sends it to the group,
who then initiate a reduction using the future as reduction target.

Note that using a reduction to know when all the group members have finished is preferable
to sending multiple point-to-point messages because, like explained earlier,
reductions are optimized to be scalable on very large systems,
and also simplify code.

This is an example of the output of Hello World running of 4 processors:

.. code-block:: text

    $ python -m charmrun.start +p4 hello_world.py ++quiet
    Hello World from element 0
    Hello World from element 2
    Hello World from element 1
    Hello World from element 3

The output brings us to an important fact:

.. note::
    For performance reasons, by default Charm does not enforce or guarantee any particular
    order of delivery of messages (remote method invocations) or order in which chare
    instances are created on remote processes. There are multiple mechanisms to sequence
    messages. The ``when`` decorator is a simple and powerful mechanism to specify
    when methods should be invoked.

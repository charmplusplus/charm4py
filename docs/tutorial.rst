============
Tutorial
============

.. contents::

Most of the code throughout this section can also be found in ``examples/tutorial``.

Program start and exit
----------------------

To start a Charm program, you need to invoke the ``charm.start()`` method.
We will begin with a simple usage pattern [#]_:

.. code-block:: python

    # examples/tutorial/start.py
    from charmpy import charm

    def main(args):
        print("Charm program started on processor", charm.myPe())
        print("Running on", charm.numPes(), "processors")
        charm.exit()

    if __name__ == '__main__':
        charm.start(entry=main)

We need to define an entry point to the Charmpy program, which we refer to as the
Charm *main* function.
In our example it is the function called ``main`` .
The main function runs on only one processor, typically processor 0, and is in charge
of creating and distributing work across the system. The main function must take
one argument to get the list of command-line arguments.
In this example, we are specifying the
function ``main`` as the main function by passing it to ``start`` method using the
``entry`` keyword argument.

The method ``numPes`` returns the number of processors (aka Processing Elements) on
which the distributed program is running. The method ``myPe`` returns the processor
number on which the caller resides.

``charm.exit()`` is called to exit a Charm program. It can be called from any chare
on any processor.

.. [#] More advanced use cases like what to do if Chares are defined in multiple
       modules are discussed in the manual.

Defining Chares
---------------

To define a Chare, simply define a class that is a subclass of ``Chare``.

.. code-block:: python

    from charmpy import Chare

    class MyChare(Chare):
        def __init__(self):
            pass

        def work(self, data):
            # ... do something ...

Methods of ``MyChare`` will be remotely callable by other chares.

For easy management of distributed objects, the user can create distributed collections:


.. code-block:: python

    # examples/tutorial/chares.py
    from charmpy import charm, Chare, Group, Array

    class MyChare(Chare):
        def __init__(self):
            print("Hello from MyChare instance in processor", charm.myPe())

        def work(self, data):
          pass

    def main(args):

        # create one instance of MyChare on every processor
        my_group = Group(MyChare)

        # create 3 instances of MyChare, distributed among all cores by the runtime
        my_array = Array(MyChare, 3)

        # create 2 x 2 instances of MyChare, indexed using 2D index and distributed
        # among all cores by the runtime
        my_2d_array = Array(MyChare, (2, 2))

    if __name__ == '__main__':
        charm.start(entry=main)

The above program will create P + 3 + 2\*2 chares and print a message for each created
chare, where P is the number of processors used to launch the program.

This is the output for 2 PEs:

.. code-block:: text

    $ ./charmrun +p2 /usr/bin/python3 examples/tutorial/chares.py ++local ++quiet
    Hello from MyChare instance in processor 0
    Hello from MyChare instance in processor 0
    Hello from MyChare instance in processor 0
    Hello from MyChare instance in processor 0
    Hello from MyChare instance in processor 0
    Hello from MyChare instance in processor 1
    Hello from MyChare instance in processor 1
    Hello from MyChare instance in processor 1
    Hello from MyChare instance in processor 1

If running the example, note that it will not exit because a suitable exit point has
not been defined (more on this below). For now, press CTRL-C to exit.

.. note::
    Chares can only be created once the Charm *main* function has been reached.

Distributed method invocation
-----------------------------

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

By default, method invocation is *asynchronous*, i.e. it returns immediately without
waiting for the actual method to be invoked on the remote object, and therefore
without returning any result. Asynchronous method invocation is desirable in many
cases because it leads to better overlap of computation and communication and better
resource utilization (which translates to more speed). Note that this does not mean
that we cannot obtain a result from a remote chare as a result of calling
one of its methods. For example:

.. code-block:: python

    # --- in chare 0 ---
    def doWork(self):
        group[1].apply(3) # tell chare 1 to apply 3 to its data, returns immediately

    def storeResult(self, data):
        # got resulting data from remote object
        # do something with data

    # --- in chare 1 ---
    def apply(self, x):
      self.data += x  # apply parameter
      group[0].storeResult(self.data.copy())  # return result to caller

However, the user can also invoke methods synchronously if desired (e.g. to more
conveniently wait for a result) by using the keyword ``block``:

.. code-block:: python

    # wait for a value from chare with index (10,10)
    x = my_array[10,10].apply(3, block=True)

Proxies can be sent to other chares as arguments of methods. We will see this in
the *Hello World* example below.

Reductions 101
--------------

Reductions can be performed by members of a collection with the result being sent to
any chare of your choice.

.. code-block:: python

    # examples/tutorial/reduction.py
    from charmpy import charm, Chare, Group, Reducer

    class MyChare(Chare):
        def __init__(self):
          pass

        def work(self, data):
            self.contribute(data, Reducer.sum, self.thisProxy[0].collectResult)

        def collectResult(self, result):
            print("Result is", result)
            charm.exit()

    def main(args):
        my_group = Group(MyChare)
        my_group.work(3)

    if __name__ == '__main__':
        charm.start(entry=main)


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
operations that are applied to the data are called *reducers*, and Charmpy includes
several built-in reducers (including ``sum``, ``max``, ``min``, ``product``, ``gather``),
as well as allowing users to define their own custom reducers for use in reductions. Please
refer to the manual for more information.

Arrays (array.array_) and `NumPy arrays`_ can be passed as contribution to many of
Charmpy's built-in reducers. The reducer will be applied to elements
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

Now we will show a full *Hello World* example:

.. code-block:: python

    # examples/tutorial/hello_world.py
    from charmpy import Chare, Mainchare, Group, charm

    class Main(Mainchare):

        def __init__(self, args):
            # create Group of Hello objects (one object exists and runs on each core)
            hellos = Group(Hello)
            # call method 'SayHello' of all group members, passing proxy to myself
            hellos.SayHi(self.thisProxy)

        # called when every element has contributed
        def done(self):
            charm.exit()

    class Hello(Chare):

        def __init__(self):
            pass

        def SayHi(self, main):
            print("Hello World from element", self.thisIndex)
            # contribute to empty reduction to end program
            self.contribute(None, None, main.done)

    charm.start()

This program prints a "Hello World" message from all processors.

Here we introduce a new type of chare called ``Mainchare``. A Mainchare constructor
serves as an Charm *main* function. A Mainchare is also frequently
used as a program exit point. An instance of ``Mainchare`` is a chare that exists only on PE 0.

The Mainchare requests the creation of a ``Group`` of chares of type ``Hello``.
Here it is important to note that group creation is asynchronous and as
such the chares in the group have not been created yet when the call returns.
It then tells all the members of the group to say hello, also passing a proxy to
itself (``self.thisProxy``).

When the method is invoked on the remote chares, they print their message along
with their index in the group (which is stored in the attribute ``thisIndex``).
For groups, the index coincides with the PE number.

We want to exit the program only after all the chares have printed their message.
However, since they reside on different processes, we need to communicate this
fact to a central point.
To know when they have concluded,
we could have each of them individually send a message to ``main`` using its proxy.
However, we use an "empty" reduction (with no data) instead. A reduction is preferable
because, like explained earlier, they are optimized to be scalable on very large systems,
and also because it removes bookkeeping burden from the programmer, as the target
receives only one method invocation as opposed to N, where N is the number of elements
in the collection.


This is an example of the output running of 4 processors:

.. code-block:: text

    $ ./charmrun +p4 /usr/bin/python3 examples/tutorial/hello_world.py ++local ++quiet
    Hello World from element 0
    Hello World from element 2
    Hello World from element 1
    Hello World from element 3

The output brings us to an important fact:

.. note::
    By default, Charm does not enforce or guarantee any particular order of delivery of messages
    (remote method invocations) or order in which chare instances are created on remote
    processes. There are multiple mechanisms to sequence messages, like using the
    ``when`` decorator or by including an identifier as part of a method invocation
    to sequence message processing.

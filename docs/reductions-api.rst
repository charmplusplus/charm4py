
Reductions
==========

A reduction is a distributed and scalable operation that reduces data
distributed across chares into a smaller set of data.
A reduction involves the chares in a collection (Group, Array or Section). They are
started by the elements calling their ``Chare.reduce()`` or ``Chare.allreduce()``
methods (see :ref:`Chare <chare-api-label>`).

.. important::

    Reduction calls are asynchronous and return immediately. Chares can start
    multiple reduction operations at the same time, but every chare in the same
    collection must contribute to reductions in the same order.


.. _reducer-api-label:

Reducers
--------

``charm4py.Reducer`` contains the reducer functions that have been registered with
the runtime. Reducer functions are used in reductions, to aggregate data across the members
of a chare collection.
``Reducer`` has the following built-in attributes (reducers) for use in reductions:

* ``max``: max function. When contributions are vectors (lists or arrays) of numbers,
  the reduction result will be the pairwise or "parallel" maxima of the vectors.

* ``min``: min function. Pairwise minima in the case of vector contributions.

* ``sum``: sum function. Pairwise sum in the case of vector contributions.

* ``product``: product function. Pairwise product in the case of vector contributions.

* ``logical_and``: logical and. Requires bool values or arrays of bools.

* ``logical_or``: logical or. Requires bool values or arrays of bools.

* ``logical_xor``: logical xor. Requires bool values or arrays of bools.

* ``gather``: Adds contributions to a Python list, and sorts the list based
  on the index of the contributors in their collection.


Registering custom reducers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To register a custom reducer function:

* **Reducer.addReducer(func, pre=None, post=None)**:

    Where *func* is a Python function with one parameter (list of contributions),
    and must return the result of reducing the given contributions.
    *pre* is optional and is a function intended to pre-process data passed in
    reduce calls.
    It must take two parameters ``(data, contributor)``, where ``data`` is
    the data passed in a reduce call and ``contributor`` is the chare object.
    *post* is optional and is a function intended to post-process the data after the whole
    reduction operation has completed. It takes only one parameter, which is the reduced data.

To refer to a custom reducer:

``Reducer.name``, where ``name`` is the name of the function that was passed to ``addReducer``.


Example
-------

.. code-block:: python

    from charm4py import charm, Chare, Group, Reducer
    import numpy as np

    DATA_SIZE = 100
    NUM_ITER = 20

    class A(Chare):

        def __init__(self):
            self.data = np.zeros(DATA_SIZE)
            self.iteration = 0

        def work(self):
            # ... do some computation, modifying self.data ...
            # do reduction and send result to element 0
            self.reduce(self.thisProxy[0].collectResult, self.data, Reducer.sum)

        def collectResult(self, result):
            # ... do something with result ...
            self.iteration += 1
            if self.iteration == NUM_ITER:
                exit()
            else:
                # continue doing work
                self.thisProxy.work()

    def main(args):
        g = Group(A)
        g.work()

    charm.start(main)

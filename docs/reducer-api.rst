
.. _reducer-api-label:

Reducer
-------

``charm4py.Reducer`` contains the reducer functions that have been registered with
the runtime. Reducer functions are used in Reductions, to aggregate data across the members
of a chare collection (see :ref:`chare-api-label`).

Reducers
~~~~~~~~

``Reducer`` has the following built-in attributes (reducers) for use in reductions:

* ``max``: max function. When contributions are vectors (lists or arrays) of numbers,
  the reduction result will be the pairwise or "parallel" maxima of the vectors.

* ``min``: min function. Pairwise minima in the case of vector contributions.

* ``sum``: sum function. Pairwise sum in the case of vector contributions.

* ``product``: product function. Pairwise product in the case of vector contributions.

* ``nop``: This is used for empty reductions which do not contribute any data. Empty
  reductions are useful to know when all the chares in a collection have reached
  a certain point (e.g. synchronization purposes).
  Passing ``None`` as reducer in ``contribute`` calls has the same effect.

* ``gather``: Adds contributions to a Python list, and sorts the list based
  on the index of the contributors in their collection.


Registering custom reducers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To register a custom reducer function:

``Reducer.addReducer(func, pre=None, post=None)``

where ``func`` is a Python function with one parameter (list of contributions),
and must return the result of reducing the given contributions.
``pre`` is optional and is a function intended to pre-process data passed in ``Chare.contribute()`` calls.
It must take two parameters ``(data, contributor)``, where ``data`` is
the data passed in a contribute call and ``contributor`` is the chare object.
``post`` is optional and is a function intended to post-process the data after the whole
reduction operation has completed. It takes only parameter which is the reduced data.

To refer to a custom reducer:

``Reducer.name``, where ``name`` is the name of the function that was passed to ``addReducer``.

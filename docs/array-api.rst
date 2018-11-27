
.. _array-api-label:

Array
-----

``charm4py.Array`` is a type of distributed collection where chares have
n-dimensional indexes (represented by an integer n-tuple), and members can exist
anywhere on the system. As such, there can
be zero or multiple elements of a chare array on a given PE, and elements can
migrate between PEs.

Arrays are created using the following syntax:

``charm4py.Array(chare_type, dims=None, ndims=-1, args=[], map=None)`` where
``chare_type`` is the type of chares that will constitute the array.
There are two modes to create an array:

    1. Specifying the bounds. ``dims`` is an n-tuple indicating the size of
       each dimension. The number of elements that will be created is the product
       of the sizes of every dimension. For example, ``dims=(2,3,5)`` will create
       an array of 30 chares with 3D indexes.

    2. Empty array of unspecified bounds, when ``dims=None``. ``ndims`` indicates
       the number of dimensions to be used for indexes.

    ``args`` is the list of arguments to pass to the constructor of each element.
    ``map`` can be optionally used to specify an ``ArrayMap`` for initial mapping
    of chares to PEs (see below). It must be a proxy to the map. If unspecified,
    the system will choose a default mapping.


The call to create an array returns a proxy to the array.

Any number of arrays (of the same or different chare types) can be created. Each
array that is created has a unique integer identifier, called the "Array ID".

.. note::
    The call to create an array returns immediately without waiting for all the
    elements to be created. See ``charm.awaitCreation()`` for one mechanism to wait
    for creation.

.. important::
    Arrays with unspecified bounds support dynamic insertion of elements via the
    array proxy (see below). Note that these types
    of arrays can be sparse in the sense that elements need not have contiguous
    indexes. Elements can be inserted in any order, from any location, at any time.

Array Proxy
~~~~~~~~~~~

An Array proxy references a chare array and its elements. An array proxy is returned
when creating an Array (see above) and can also be accessed from the attribute ``thisProxy``
of the elements of the array (see :ref:`chare-api-label`). Like any proxy, array proxies
can be sent to *any* chares in the system.

Attributes
++++++++++

* **aid**: The ID of the array that the proxy references.

* **ndims**: Number of dimensions of the indexes used by the array.

* **elemIdx**: This is an empty tuple if the proxy references the whole array, otherwise
  it is the index of an individual element in the array.

Methods
+++++++

* **self[index]**: return a new proxy object which references the element in the array
  with the given ``index``.

* **self.ckInsert(index, args=[], onPE=-1)**: Insert an element with ``index`` into
  the array. This is only valid for arrays that were created empty (with unspecified
  bounds). ``args`` is the list of arguments passed to the constructor of the element.
  ``onPE`` can be used to indicate on which PE to create the element.

* **self.ckDoneInserting()**: This must be used when finished adding elements with
  ``ckInsert``.


ArrayMap
~~~~~~~~

An ``ArrayMap`` is a special type of Group whose function is to customize the initial
mapping of chares to PEs for a chare Array.

A custom ArrayMap is defined by writing a new class that inherits from ``ArrayMap``,
and defining the method ``procNum(self, index)``, which receives the index of an array element,
and returns the PE number where that element must be created.

To use an ArrayMap, it must first be created like any other Group, and the proxy to the
map must be passed to the Array constructor (see above).

Note that array elements may migrate after creation and the ArrayMap only determines
the initial placement.


Chare Collections
=================

.. _group-api-label:

Group
-----

``charm4py.Group`` is a type of collection where there is one chare per PE.
These chares are not migratable and are always bound to the PE where they are created.
Elements in groups are indexed by integer ID, which for each element coincides with the
PE number where it is located.

Groups are created using the following syntax:

* **Group(chare_type, args=[], onPEs=None)**:

    Where *chare_type* is the
    type of chares that will constitute the group. The list of arguments to pass
    to the constructor of each element is given by *args*.
    If *onPEs* is None, creates one instance on every PE and returns a group proxy.
    If *onPEs* is a list of PEs, only creates instances on those PEs, and returns a section proxy.
    In this case, the ``thisProxy`` member of each instance will be a section proxy,
    and broadcasts/reductions will only involve the specified PEs.

    Any number of groups (of the same or different chare types) can be created.

    .. note::
        The call to create a Group is asynchronous: it returns immediately without waiting for the
        elements to be created. See ``charm.awaitCreation()`` for one mechanism to wait
        for creation.


Group Proxy
~~~~~~~~~~~

A Group proxy references a chare group and its elements. A group proxy is returned
when creating a Group (see above) and can also be accessed from the attribute ``thisProxy``
of the elements of the group. Like any proxy, group proxies
can be sent to *any* chares in the system.


Methods
+++++++

* **self[index]**: return a new proxy object which references the element in the group
  with the given *index*.

Group proxies can be **sliced** to obtain section proxies (see :doc:`sections`).


.. _nodegroup-api-label:

NodeGroup
---------

``charm4py.NodeGroup`` is a collection with one chare per logical node, that is, per
process. Its elements (branches) are indexed by node number, from 0 to
``charm.numNodes() - 1``; inside a branch, ``self.thisIndex`` is ``charm.myNode()``.
With one PE per process, which is what Charm4py runs today, node numbers equal PE
numbers, but a program should use ``charm.myNode()`` and ``charm.numNodes()`` and not
rely on that. Like group elements, branches are not migratable.

* **NodeGroup(chare_type, args=[])**:

    Creates one instance of *chare_type* (a subclass of ``Chare``) on every node,
    passing *args* to each constructor, and returns a nodegroup proxy. As for
    groups, the call is asynchronous (see ``charm.awaitCreation()``).

Every registered chare class can be used for a NodeGroup. A class with an entry
method marked ``@entry(exclusive=True)`` or ``@entry(immediate=True)`` (see
:ref:`entry-attributes-label`) is registered for NodeGroup only; using it as a Group,
Array or single chare, or registering it explicitly for those, raises
``Charm4PyError``.

Branches contribute to reductions with ``self.contribute``, ``self.reduce`` and
``self.allreduce`` as group elements do; the reduction is over the branches of the
nodegroup. ``charm.checkpoint`` saves nodegroups with the other chares, and a
restarted program finds them restored. The branches on this process are in
``charm.nodegroups`` (nodegroup id to branch), separate from ``charm.groups``.

NodeGroup Proxy
~~~~~~~~~~~~~~~

A nodegroup proxy is returned when creating a NodeGroup and is the ``thisProxy`` of
its branches. Calling a method on it broadcasts to every branch; with ``ret=True``
the future receives the return values in node order.

* **self[node]**: return a new proxy object which references the branch on node
  *node*.
* **self.ckLocalBranch()**: the branch on the caller's node.

Nodegroup proxies cannot be sliced, and nodegroups have no sections.


.. _array-api-label:

Array
-----

``charm4py.Array`` is a type of distributed collection where chares have
n-dimensional indexes (represented by an integer n-tuple), and members can exist
anywhere on the system. As such, there can
be zero or multiple elements of a chare array on a given PE, and elements can
migrate between PEs.

Arrays are created using the following syntax:

* **Array(chare_type, dims=None, ndims=-1, args=[], map=None, useAtSync=False)**:

    Where *chare_type* is the type of chares that will constitute the array.
    There are two modes to create an array:

      1. Specifying the bounds. *dims* is an n-tuple indicating the size of
         each dimension. The number of elements that will be created is the product
         of the sizes of every dimension. For example, ``dims=(2, 3, 5)`` will create
         an array of 30 chares with 3D indexes. If it is a 1D array, *dims*
         can also be the number of elements in the array.

      2. Empty array of unspecified bounds, when ``dims=None``. *ndims* indicates
         the number of dimensions to be used for indexes.

    The list of arguments to pass to the constructor of each element is given
    by *args*. *map* can optionally be used to specify an ``ArrayMap`` for initial
    mapping of chares to PEs (see below). It must be a proxy to the map. If unspecified,
    the system will choose a default mapping. If the elements of this array
    will use AtSync for load balancing *useAtSync* must be ``True`` (see
    :ref:`Chare <chare-api-label>`).

    The call to create an array returns a proxy to the array.

    Any number of arrays (of the same or different chare types) can be created.

    .. Each array that is created has a unique integer identifier, called the "Array ID".

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
of the elements of the array. Like any proxy, array proxies
can be sent to *any* chares in the system.


Methods
+++++++

* **self[index]**: return a new proxy object which references the element in the array
  with the given *index*.

* **self.ckInsert(index, args=[], onPE=-1)**: Insert an element with *index* into
  the array. This is only valid for arrays that were created empty (with unspecified
  bounds). *args* is the list of arguments passed to the constructor of the element.
  *onPE* can be used to indicate on which PE to create the element.

* **self.ckDoneInserting()**: This must be used when finished adding elements with
  ``ckInsert``.

* **self.ckDestroy(ret=False)**: On an element proxy (``proxy[index].ckDestroy()``),
  send a message that destroys that element wherever it lives (Charm++
  ``ArrayElement::ckDestroy``). With ``ret=True`` it returns a future that receives
  ``None`` once the element is gone. On the whole array (``proxy.ckDestroy()``) or a
  section it destroys every member; ``ret`` and ``awaitable`` raise
  ``Charm4PyError`` there, because the completion of a broadcast is a reduction over
  the members, which no longer exist. The element leaves ``charm.arrays`` on its PE,
  and reductions over the array count only the remaining elements. A message sent
  later to a destroyed index is not delivered and is not an error: as in Charm++,
  the runtime keeps it for a later ``ckInsert`` of the same index, which then
  receives it (a future waiting on such a message never completes). An element can
  also destroy itself with ``self.ckDestroy()``; see :doc:`chare-api`.

Array proxies can be **sliced** to obtain section proxies (see :doc:`sections`).

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

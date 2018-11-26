
.. _group-api-label:

Group
-----

``charm4py.Group`` is a type of collection where there is one chare per PE.
These chares are not migratable and are always bound to the PE where they are created.
Elements in groups are indexed by integer ID, which for each element coincides with the
PE number where it is located.

Groups are created using the following syntax:

``charm4py.Group(chare_type, args=[])`` where ``chare_type`` is the type of chares
that will constitute the group. ``args`` is the list of arguments to pass to the
constructor of each element.

The call to create a group returns a proxy to the group.

Any number of groups (of the same or different chare types) can be created. Each
group that is created has a unique integer identifier, called the "Group ID".

.. note::
    The call to create a Group returns immediately without waiting for all the
    elements to be created. See ``charm.awaitCreation()`` for one mechanism to wait
    for creation.

Group Proxy
~~~~~~~~~~~

A Group proxy references a chare group and its elements. A group proxy is returned
when creating a Group (see above) and can also be accessed from the attribute ``thisProxy``
of the elements of the group (see :ref:`chare-api-label`). Like any proxy, group proxies
can be sent to *any* chares in the system.

Attributes
++++++++++

* **gid**: The ID of the group that the proxy references.

* **elemIdx**: This is ``-1`` if the proxy references the whole group, otherwise it is the
  index of an individual element in the group.

Methods
+++++++

* **self[index]**: return a new proxy object which references the element in the group
  with the given ``index``.

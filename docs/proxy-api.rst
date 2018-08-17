
.. _proxy-api-label:

Proxy
-----

Proxy classes do not exist a priori. They are generated at runtime using metaprogramming,
based on the definition of the chare types that are registered when the runtime is started.

Proxy objects are returned when creating chares or collections, and are also stored
in the ``thisProxy`` attribute of chares.

.. tip::
    A proxy object can be sent to any chare(s) in the system via remote methods.

Proxies have the same methods as the chare that they reference.
Calling those methods will result in the method being invoked on the chare(s) that
the proxy references, regardless of the location of the chare.

The syntax to call a remote method is:

**proxy.remoteMethod(*args, ret=False)**:

  Calls the method of the chare(s) referenced
  by the proxy. This represents a remote method invocation. If the proxy references a
  collection, a broadcast call is made and the method is invoked on all chares
  in the collection. Otherwise, the method is called on an individual chare.
  The call returns immediately and does not wait for the method to be invoked at the
  remote chare(s).
  If the optional keyword argument ``ret`` is ``True``, this returns a :ref:`Future <futures-api-label>`,
  which can be used to wait for the result. This also works for broadcast calls. In this
  case, the return value will be ``None`` and will be returned when the method has
  been invoked on every element.

Proxies that reference a Group/Array or its elements have additional properties (see :ref:`group-api-label`
and :ref:`array-api-label`).

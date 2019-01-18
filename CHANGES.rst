
This describes the most significant changes. For more detail, see the commit
log in the source code repository.

What's new in v0.12
===================

* Added experimental charm.pool library which is similar to Python's
  multiprocessing Pool, but also works in a distributed setting (multiple hosts),
  tasks can create other tasks all of which use the same shared pool,
  and can benefit from Charm++'s support for efficient communication layers
  such as MPI. See documentation for more information.

* Improved support for building and running with Charm++'s MPI communication
  layer. See Install and Running sections of the documentation for more information.

* Substantially improved the performance of threaded entry methods by allowing
  thread reuse.

* Blocking allreduce and barrier is now supported inside threaded entry methods:
  ``result = charm.allReduce(data, reducer, self)`` and ``charm.barrier(self)``.

* Can now indicate if array elements use AtSync at array creation time
  by passing ``useAtSync=True`` in Array creation method.

* Minor bugfixes and improvements.


What's new in v0.11
===================

* Changed the name of the project from CharmPy to *charm4py* (more information on why
  we changed the name is in the forum).

* Not directly related to this release, but there is a new forum for charm4py discussions
  (see contact details). Feel free to visit the forum for discussions, reports,
  provide feedback, request features and to follow development.

* Support for interactive charm4py shell using multiple processes on one host has been added
  as a *beta* feature. Please provide feedback and suggestions in the forum or GitHub.

* Uses the recent major release of Charm++ (6.9)

* C-extension module can be built on Windows. Windows binary wheels on PyPI come with
  the compiled extension module.

* API change: method ``Chare.gather()`` has been removed to make the name available
  for user-defined remote methods. Use ``self.contribute(data, Reducer.gather, ...)``
  instead.

* Some methods of ``charm`` are now remotely callable, like ``charm.exit()``.
  They can be used as any other remote method including as targets of reductions.
  For example: ``self.contribute(None, None, charm.thisProxy[0].exit)``

* Can now use Python exit function instead of ``charm.exit()``

* Other small fixes and improvements.


What's new in v0.10.1
=====================

This is a bugfix and documentation release:

* Added core API to docs, and more details regarding installation and running

* Fixed reduction to Future failing when contributing numeric arrays

* Charm4py now requires Charm++ version >= ``6.8.2-890`` which, among other things,
  includes fixes for the following Windows issues:

      - Running an application without ``charmrun`` on Windows would crash

      - Abort messages were sometimes not displayed on exit. On Charm4py,
        this had the effect that Python runtime errors were sometimes not shown.

      - If running with charmrun, any output prior to charm.start()
        would not be shown. On Charm4py, this had the effect that Python
        syntax errors were not shown.


What's new in v0.10
===================

**Installation and Setup**

* Charm4py can be installed with pip (``pip install charm4py``) on regular
  Linux, macOS and Windows systems

* Support setuptools to build, install, and package Charm4py

* Installation from source is much simpler (see documentation)

* charm4py builds include the charm++ library and are relocatable. ``LD_LIBRARY_PATH`` or
  similar schemes are no longer needed.

* charm4py does not need a configuration file anymore (it will automatically
  select the best available interface layer at runtime).


**API Changes**

* Start API is now ``charm.start(entry)``, where ``entry`` can be a regular
  Python function, or any chare type. Special Mainchare class is no longer needed.


**Performance**

* Added Cython-based C-extension module to considerably speed up the interface with
  the Charm++ library and critical parts of charm4py (currently only with Python 3+).

* Several minor performance improvements


**Features**

* *Threaded entry methods*: entry methods can run in their own thread when tagged
  with the ``@threaded`` decorator. This enables `direct style programming`__ with
  asynchronous remote method execution (also see Futures):

    - The entry point (main function or chare) is automatically threaded by default

    - Added ``charm.awaitCreation(*proxies)`` to wait for Group and Array creation
      within the threaded entry method that created them

    - Added ``self.wait('condition')`` construct to suspend entry method execution until a condition is
      met

* *Futures*

    - Remote method invocations can optionally return futures with the ``ret``
      keyword: ``future = proxy.method(ret=True)``. Also works for broadcasts.
    - A future can be queried to obtain the value with ``future.get()``. This will
      block if the value has not yet been received.
    - Futures can be explicitly created using ``future = charm.createFuture()``,
      and passed to other chares. Chares can send values to the future by calling
      ``future.send(value)``
    - Futures can be used as reduction targets

* Simplified ``@when`` decorator syntax and enhanced to support general conditions
  involving a chare's state and remote method arguments. New syntax is ``@when('condition')``.

* Can now pass arguments to chare constructors

* Can create singleton chares. Syntax is ``proxy = Chare(MyChare, pe)``

* ArrayMap: to customize initial mapping of chares to cores

* Warn if user forgot to call ``charm.start()`` when launching charm4py programs

* Exposed ``migrateMe(toPe)`` method of chares to manually migrate a chare to indicated
  PE

* Exposed `LBTurnInstrumentOn/Off`__ from Charm++ to charm4py applications

* Interface to construct topology-aware trees of nodes/PEs


**Bug Fixes**

* Fixed issues related to migration of chares


**Documentation**

* Updated documentation and tutorial to reflect changes in installation, setup,
  addition of Futures and API changes

* Added leanmd results to benchmarks section


**Examples and Tests**

* Improved performance of ``stencil3d_numba.py``, and added better benchmarking support
* Added parallel map example (``examples/parallel-map/parmap.py``)
* Improved output and scaling of several tests when launched with many (> 100)
  PEs
* Cleaned, updated, simplified several tests and examples by using futures


**Profiling**

* Fixed issues which resulted in inaccurate timings in some circumstances
* Profiling of chare constructors (including main chare and chares that
  are migrating in) is now supported


**Code**

* Code has been structured as a Python package

* Heavy code refactoring. Code simplification in several places

* Several improvements towards PEP 8 compliance of core charm4py code.
  Indentation of code in ``charm4py`` package is PEP 8 compliant.

* Improvements to test infrastructure and added Travis CI script


.. __: https://en.wikipedia.org/wiki/Direct_style
.. __: http://charm.cs.illinois.edu/manuals/html/charm++/7.html#SECTION01650000000000000000


What's new in v0.9
==================

**General**

* Charm4py is compatible with Python 3 (Python 3 is the recommended option)

* Added documentation (http://charm4py.readthedocs.io)


**API Changes**

* New API to create chares and collections:
  all chare types are defined by inheriting from Chare.
  To create a group: ``group_proxy = Group(MyChare)``.
  To create an array: ``array_proxy = Array(MyChare, ...)``.

* Simplified program start API with automatic registration of chares


**Performance**

* Bypass pickling of common array types (most notably numpy arrays) by directly
  copying contents of their buffer into messages. This can result in substantial
  performance improvement.

* Added optional CFFI-based layer to access Charm++ library, that is faster than
  existing ctypes layer.

* The ``LOCAL_MSG_OPTIM`` option (True by default) avoids copying and serializing
  messages that are directed to an object in the same process. Works for all chare
  types.


**Features**

* Support reductions over chare arrays/groups, including defining custom reducers.
  Numpy arrays and numbers can be passed as data and will be efficiently reduced.
  Added "gather" reducer.

* Support dynamic insertion into chare arrays

* Allow using int as index of 1D chare array

* ``element_proxy = proxy[index]`` syntax now returns a new independent proxy object to
  an individual element

* Added ``@when('attrib_name')`` decorator to entry methods so that they are invoked
  only when the first argument matches the value of the specified chare's attribute


* Added methods ``charm.myPe()``, ``charm.numPes()``, ``charm.exit()`` and
  ``charm.abort()`` as alternatives to CkMyPe, CkNumPes, CkExit and CkAbort


**Other**

* Improved profiling output. Profiling is disabled by default.

* Improved general error handling and output. Errors in charm4py runtime raise
  ``Charm4PyError`` exception.

* Code Examples:

    - Updated stencil3d examples to use the ``@when`` construct

    - Added particle example (uses the ``@when`` construct)

    - Add total iterations as program parameter for wave2d

* Added ``auto_test.py`` script to test charm4py

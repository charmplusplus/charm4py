============
Running
============

.. .. contents::

Prerequisites
-------------

1. Make sure that the Charm++ shared library can be found
   and loaded by the system's dynamic linker before running Charmpy programs.

   On Linux there are multiple ways to accomplish this. One way is::

   $ export LD_LIBRARY_PATH=/path/to/libcharm_dir

   On macOS, try the equivalent ``DYLD_LIBRARY_PATH``, or place or symlink the library on a
   system directory, like ``/usr/local/lib``. More macOS information is in the troubleshooting section below.

   If using the ``ctypes`` layer, verify that the path to the shared library is
   correct in ``charmpy.cfg``.

2. The Charmpy module (``charmpy.py``) must be in the Python search path. One way to
   set it is::

   $ export PYTHONPATH=path_to_charmpy_folder  # NOTE: the folder containing charmpy.py

Launching Charmpy programs
--------------------------

There are many example programs included with this distribution.
To launch Charmpy programs on a Linux or Mac desktop, use the ``charmrun`` binary
that was generated when building Charm++::

  $ ./charmrun +p4 /usr/bin/python3 examples/hello/group_hello.py ++local

The charmrun option ``+pN`` specifies how many processors to run the program with. It
will launch one process per processor.

Charmpy programs accept the `same command-line parameters`_ as Charm++.

charmrun can also be used to launch a program on `a network of workstations`_.

.. _a network of workstations: http://charm.cs.illinois.edu/manuals/html/charm++/C.html#SECTION05330000000000000000

.. _same command-line parameters: http://charm.cs.illinois.edu/manuals/html/charm++/C.html


Troubleshooting
---------------

Issue
    I get these or similar errors when launching a Charmpy program::

    ... libcharm.so: undefined symbol: __executable_start
    ... libcharm.so: undefined symbol: CkRegisterMainModule

Solution
    Please make sure that you have built Charm++ with these flags:
    ``--build-shared --enable-charmpy``
    See :doc:`install` for more information.

|


Issue
    Program hangs with no output when launching with ``charmrun``.

Solution
    - If running **with** ``++local``:

      Please make sure that you have built Charm++ with these flags:
      ``--build-shared --enable-charmpy``
      and that you are using the version of ``charmrun`` that is generated
      with the above flags.
      See :doc:`install` for more information.

    - If running **without** ``++local``:

      First make sure Charm++ was built as specified above.

      Second make sure that necessary environment variables (like PYTHONPATH
      and LD_LIBRARY_PATH) are being exported to remote processes. To do this,
      write the necessary export commands in ``$HOME/.charmrunrc``

      If problems persist, this typically indicates that an error ocurred in
      the program before handling control to libcharm (e.g. a Python syntax
      error). Please run with ``++local`` to debug.

Running with CFFI interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Solutions for some issues observed in macOS environment:

  1. To load the Charm++ shared library through the dynamic linker without turning off SIP, the
     following symlink solution can be used::

     $ ln -s /path/to/libcharm.so /usr/local/lib

  2. When executing a Charmpy program with the CFFI interface, if an error related
     to unsafe use of relative rpath libcharm.so appears, try the following::

     $ cd charmpy/__cffi_objs__
     $ install_name_tool -change libcharm.so /usr/local/lib/libcharm.so _charmlib.so

     Note: This assumes that you have symlinked the library as indicated in (1).


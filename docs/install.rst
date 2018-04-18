============
Install
============

.. contents::

Requirements
------------

  - Python 2.7 or 3.3+. Python 3 is *highly* recommended for best performance, and
    for continued support.
    Charmpy has been tested with the following implementations: CPython (most common
    Python implementation) and PyPy_.

  - `Charm++`_ shared library. See below for instructions on building the shared library.

  - (Optional, but *recommended*): Cython_ allows compilation of parts of the runtime,
    and is highly recommended for best performance.

  - Charmpy can run on the wide variety of systems supported by Charm++, which includes
    many supercomputers in the TOP500.
    To date it has been tested on Linux (including the `Windows Subsystem for Linux`_),
    macOS and Cray XC/XE.

.. _Cython: http://cython.org/

.. _PyPy: http://pypy.org

.. _Charm++: http://charmplusplus.org/

.. _Windows Subsystem for Linux: https://docs.microsoft.com/en-us/windows/wsl/about

Download Charmpy
----------------

Get the Charmpy source code here_.

.. _here: https://github.com/UIUC-PPL/charmpy

Building the Charm++ shared library
-----------------------------------

First download Charm++::

    $ git clone https://charm.cs.illinois.edu/gerrit/charm
    $ cd charm

Next, build the library. Below are instructions for regular Linux and macOS
environments. These generally apply to Linux clusters as well.
On Unix-like environments, you will need these packages: ``autoconf``, ``automake``,
and a C++ compiler.

Linux::

    $ ./build charm++ netlrts-linux-x86_64 -j4 --with-production --build-shared --enable-charmpy
    $ cd lib
    $ gcc -shared -o libcharm.so -Wl,--whole-archive libck.a libconv-core.a libconv-util.a \
      libmemory-default.a libconv-machine.a libthreads-default.a libconv-partition.a libtmgr.a \
      libhwloc_embedded.a libldb-rand.a libconv-ldb.a libmoduleGreedyRefineLB.a -Wl,--no-whole-archive -lstdc++

macOS::

    $ ./build charm++ netlrts-darwin-x86_64 -j4 --with-production --build-shared --enable-charmpy
    $ cd lib
    $ gcc -shared -o libcharm.so -Wl,-all_load libck.a libconv-core.a libconv-util.a \
      libmemory-default.a libconv-machine.a libthreads-default.a libconv-partition.a libtmgr.a \
      libhwloc_embedded.a libldb-rand.a libconv-ldb.a libmoduleGreedyRefineLB.a -Wl,-noall_load -lstdc++

.. note::
    Charm++ can be built on specialized enviroments, like Cray XE, and you can refer to the
    Charm++ manual_ and the README in Charm++'s root directory for more
    information.
    However, the last step above that generates ``libcharm.so`` may not work on some
    specialized environments. Providing
    a generic way to generate the shared library on these systems is work in progress.

.. _manual: http://charm.cs.illinois.edu/manuals/html/charm++/A.html

Updating the Charm++ shared library
-----------------------------------

If you update your version of Charmpy, you might find that it requires a more
recent version of Charm++ (libcharm). Charmpy will inform the user if this is the case.

In this situation, you can clone a recent version of Charm++ or update your git repository,
and rebuild the library (as explained in the previous section).

After this, make sure to rerun Charmpy :doc:`setup`.

Using Charmpy with Cython
-------------------------

The Cython layer is currently only supported with Python 3+.

Make sure Cython is installed for your Python distribution:
http://docs.cython.org/en/latest/src/quickstart/install.html

In addition, development files for your Python distribution may be required
(e.g. ``Python.h``) to compile the cython layer. On Ubuntu Linux, these are
usually provided by ``python3-dev`` or similar package.


Using Charmpy with CFFI
-----------------------

**CFFI version >= 1.7 required**

CFFI is the recommended layer if using PyPy (note that ``cffi`` comes builtin with PyPy).

To install a recent version, please see:
https://cffi.readthedocs.io/en/latest/installation.html
Typically, ``pip install cffi`` will suffice.

.. note::
    If you have multiple versions of Python installed (e.g. python2, python3,
    pypy), make sure CFFI is installed for the version that you will use to run Charmpy.
    For example, if you are installing CFFI with pip, make sure that you use the version of
    pip that corresponds to the interpreter you'll be using (for Python 3 it is ``pip3``
    by default on Ubuntu 16.04).

In addition, development files for your Python distribution will likely be needed
(e.g. ``Python.h``) to compile the charm library wrapper. On Ubuntu Linux, these are
usually provided by ``python3-dev`` or similar package.

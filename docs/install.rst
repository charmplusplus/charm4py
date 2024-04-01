============
Install
============

.. .. contents::

Charm4py runs on Linux, macOS, Windows, Raspberry Pi, and a wide variety of clusters and
supercomputer environments (including many supercomputers in the TOP500).

Charm4py runs on Python 3.4+. Charm4py has been tested with the
following Python implementations:
CPython (most common implementation) and PyPy_.


.. _PyPy: http://pypy.org


Manually building the Charm++ shared library
--------------------------------------------

Use this to build Charm4py binaries manually, instead of downloading prebuilt libraries from pip.
This is needed when building Charm++ for specialized machine/network layers
other than TCP and MPI (e.g. Cray XC/XE).

Before installing, you need the following prerequisites:
    - CPython: numpy, greenlet and cython (``pip3 install 'numpy>=1.10.0' cython greenlet``)
    - PyPy: none

The first step is to clone the Charm4py repository from Git::

    $ git clone https://github.com/UIUC-PPL/charm4py
    $ cd charm4py

Next, create a folder called charm_src in the charm4py repo, and then clone the Charm++ repo
into that folder::

    $ mkdir charm_src && cd charm_src
    $ git clone https://github.com/UIUC-PPL/charm

Once this is done, there are two ways to build Charm4py. The first way is to change back up
into the Charm4Py directory and run the install script::
    
    $ cd ..
    $ python3 setup.py install [--mpi]

The optional flag ``--mpi``, when enabled, will build the
Charm++ library with the MPI communication layer (MPI headers and libraries
need to be installed on the system). After this, Charm4Py will be built.

The other option is to manually build Charm++ before building Charm4py. To do this, change to
the charm directory and run the following build command::
    
    $ cd charm
    $ ./build charm4py <version> -j<N> --with-production

Then, return to the charm4py directory and run setup.py::

    $ cd ../..
    $ python3 setup.py install [--mpi]


After building, you can run Charm4py examples. One example you can try is 
array_hello.py, which can be run as follows::

    $ cd examples/hello
    $ python -m charmrun.start +p2 array_hello.py


.. note::

    The TCP layer (selected by default) will work on desktop, servers and
    small clusters. The MPI layer is faster and should work on most systems
    including large clusters and supercomputers. Charm++ however also has support
    for specialized network layers like uGNI and UCX. To use these, you have
    to manually build the Charm++ library (see below).


pip
---

This option installs prebuilt Charm4Py binaries from pip. The prebuilt pip libraries
were built with Python 3.7.

To install on regular Linux, macOS and Windows machines, do::

    $ pip3 install charm4py

.. note::

    This option uses Charm++'s TCP layer as the communication layer.
    If you want a faster communication layer (e.g. MPI), see "Install from
    source" below.

    pip >= 8.0 is recommended to simplify the install and avoid building Charm4py or
    any dependencies from sources.

    Note that a 64-bit version of Python is required to install and run Charm4py.



.. _manual: https://charm.readthedocs.io/en/latest/charm++/manual.html#installing-charm

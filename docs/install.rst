============
Install
============

.. .. contents::

Charm4py runs on Linux, macOS, Windows, Raspberry Pi, and a wide variety of clusters and
supercomputer environments (including many supercomputers in the TOP500).

Charm4py runs on Python 2.7 and 3.3+. Python 3 is *highly* recommended for best
performance and for continued support. Charm4py has been tested with the
following Python implementations:
CPython (most common implementation) and PyPy_.


.. _PyPy: http://pypy.org


pip
---

To install on regular Linux, macOS and Windows machines, do::

    $ pip3 install charm4py

.. note::

    This option uses Charm++'s TCP layer as the communication layer.
    If you want a faster communication layer (e.g. MPI), see "Install from
    source" below.

    pip >= 8.0 is recommended to simplify the install and avoid building Charm4py or
    any dependencies from sources.

    Note that a 64-bit version of Python is required to install and run Charm4py.


Install from source
-------------------

.. note::
    This is not required if installing from a binary wheel with pip.

Prerequisites:
    - CPython: numpy, greenlet and cython (``pip3 install 'numpy>=1.10.0' cython greenlet``)
    - PyPy: none

To build the latest *stable* release, do::

  $ pip3 install [--mpi] charm4py --no-binary charm4py

Or download the source distribution from PyPI, uncompress and run
``python3 setup.py install [--mpi]``.

The optional flag ``--mpi``, when enabled, will build the
Charm++ library with the MPI communication layer (MPI headers and libraries
need to be installed on the system).

To build the latest *development* version, download Charm4py and Charm++ source code
and run setup::

    $ git clone https://github.com/UIUC-PPL/charm4py
    $ cd charm4py
    $ git clone https://github.com/UIUC-PPL/charm charm_src/charm
    $ python3 setup.py install [--mpi]

.. note::

    The TCP layer (selected by default) will work on desktop, servers and
    small clusters. The MPI layer is faster and should work on most systems
    including large clusters and supercomputers. Charm++ however also has support
    for specialized network layers like uGNI and UCX. To use these, you have
    to manually build the Charm++ library (see below).


Manually building the Charm++ shared library
--------------------------------------------

This is needed when building Charm++ for specialized machine/network layers
other than TCP and MPI (e.g. Cray XC/XE).

Before running ``python3 setup.py`` in the steps above, enter the Charm++ source code
directory (``charm_src/charm``), and manually build the Charm++ library. The build
command syntax is::

    $ ./build charm4py <version> -j<N> --with-production

where ``<version>`` varies based on the system and communication layer, and ``<N>``
is the number of processes to use for compiling.
For help in choosing the correct ``<version>``, please refer to the Charm++ manual_
and the README in Charm++'s root directory.

After the library has been built, continue with ``python3 setup.py install`` in the
Charm4py source root directory.


.. _manual: https://charm.readthedocs.io/en/latest/charm++/manual.html#installing-charm

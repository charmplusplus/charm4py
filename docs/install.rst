============
Install
============

.. .. contents::

charm4py runs on Linux, macOS, Windows, and a wide variety of clusters and
supercomputer environments (including many supercomputers in the TOP500).

charm4py runs on Python 2.7 and 3.3+. Python 3 is *highly* recommended for best
performance. charm4py has been tested with the following Python implementations:
CPython (most common implementation) and PyPy_.


.. _PyPy: http://pypy.org


pip
---

To install on regular Linux, macOS and Windows machines, do::

    $ pip install charm4py

.. note::

    This option selects Charm++'s TCP layer as the communication layer.
    If you want a faster communication layer (e.g. MPI), see "Install from
    Source" below.

    pip >= 8.0 is recommended to simplify the install and avoid building charm4py or
    any dependencies from sources.

    Note that a 64-bit version of Python is required to install and run charm4py.


Install from Source
-------------------

.. note::
    This is not required if installing from a binary wheel with pip.

Prerequisites:
    - CPython: install numpy and cython (``pip install 'numpy>=1.10.0' cython``)
    - PyPy: none

To build the latest *stable* release, do::

  $ pip install [--mpi] charm4py --no-binary charm4py

Or download the source distribution from PyPI, uncompress and run
``python setup.py install [--mpi]``.

The optional flag ``--mpi``, when enabled, will build the
Charm++ library with the MPI communication layer (MPI headers and libraries
need to be installed on the system).

To build the latest *development* version, download Charm4py and Charm++ source code
and run setup::

    $ git clone https://github.com/UIUC-PPL/charm4py
    $ cd charm4py
    $ git clone https://github.com/UIUC-PPL/charm charm_src/charm
    $ python setup.py install [--mpi]

.. note::

    The TCP layer (selected by default) will work on desktop, servers and
    small clusters. The MPI layer is faster and should work on most systems
    including large clusters and supercomputers. Charm++ however also has support
    for specialized network layers like uGNI, Intel OFI and IBM PAMI. To use these,
    you have to manually build the Charm++ library (see below).


Manually building the Charm++ shared library
--------------------------------------------

This is needed when building Charm++ for specialized machine/network layers
other than TCP and MPI (e.g. Cray XC/XE).

Before running ``python setup.py`` in the steps above, enter the Charm++ source code
directory (``charm_src/charm``), and manually build the Charm++ library. The build
command syntax is::

    $ ./build charm4py <version> -j<N> --with-production

where ``<version>`` varies based on the system and communication layer, and ``<N>``
is the number of processes to use for compiling.
For help in choosing the correct ``<version>``, please refer to the Charm++ manual_
and the README in Charm++'s root directory.

After the library has been built, continue with ``python setup.py install`` in the
charm4py source root directory.


.. _manual: http://charm.cs.illinois.edu/manuals/html/charm++/A.html

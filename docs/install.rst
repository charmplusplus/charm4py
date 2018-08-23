============
Install
============

.. .. contents::

CharmPy runs on Linux, macOS, Windows, and a wide variety of clusters and
supercomputer environments (including many supercomputers in the TOP500).

CharmPy runs on Python 2.7 and 3.3+. Python 3 is *highly* recommended for best
performance. CharmPy has been tested with the following Python implementations:
CPython (most common implementation) and PyPy_.


.. _PyPy: http://pypy.org


pip
---

To install on regular Linux, macOS and Windows machines, do::

    $ pip install charmpy

.. note::
    pip >= 8.0 is recommended to simplify the install and avoid building CharmPy or
    any dependencies from sources.

    Note that a 64-bit version of Python is required to install and run charmpy.


Install from Source
-------------------

.. note::
    This is not required if installing from a binary wheel with pip (see above).

Prerequisites:
    - CPython: install numpy and cython (``pip install 'numpy>=1.10.0' cython``)
    - PyPy: none

To build the latest *stable* release, do::

  $ pip install charmpy --no-binary charmpy

Or download the source distribution from PyPI, uncompress and run ``python setup.py install``.

To build the latest *development* version, download CharmPy and Charm++ source code
and run setup::

    $ git clone https://github.com/UIUC-PPL/charmpy
    $ cd charmpy
    $ git clone https://github.com/UIUC-PPL/charm charm_src/charm
    $ python setup.py install


NOTE: the defaults used by the setup script to build Charm++ are suitable
for desktops, servers, or small clusters. Specialized environments like
supercomputers most likely require manual building with different options (see below).


Manually building the Charm++ shared library
--------------------------------------------

This is needed when building Charm++ for specialized environments (e.g. Cray XC/XE).

Before running ``python setup.py`` in the steps above, enter the Charm++ source code
directory (``charm_src/charm``), and manually build the Charm++ library. The build
command syntax is::

    $ ./build charmpy <version> -j<N> --with-production

where ``<version>`` varies based on the system and communication layer, and ``<N>``
is the number of processes to use for compiling.
For help in choosing the correct ``<version>``, please refer to the Charm++ manual_
and the README in Charm++'s root directory.

After the library has been built, continue with ``python setup.py install`` in the
charmpy source root directory.


.. _manual: http://charm.cs.illinois.edu/manuals/html/charm++/A.html

============
Install
============

.. .. contents::

Charm4py 2 runs on Charm++ with the Reconverse runtime, on Linux (x86_64 and
aarch64) and macOS (Apple Silicon). It supports the CPython implementation.
The Charm++ library it uses, ``libcharm``, and the Reconverse and LCI
libraries ``libcharm`` needs, are part of the installed package; an installed
Charm4py does not need a Charm++ source or build tree.

There are three ways to install it:

* from a wheel (`Installing a wheel`_), which needs nothing but pip;
* from source, letting the build fetch and build Charm++ itself
  (`Installing from source`_);
* from source against a Charm++ tree you build yourself
  (`Building against your own Charm++ build`_), for work on Charm++ or
  Charm4py.

Installing a wheel
------------------

Wheels exist for CPython 3.11, 3.12 and 3.13 on

* Linux x86_64 and aarch64 (``manylinux_2_28``: glibc 2.28 or newer), and
* macOS 14 or newer on Apple Silicon (arm64).

::

    $ pip install charm4py

A wheel includes ``libcharm``, Reconverse, LCI, and a libfabric built with only
its ``tcp``, ``sockets``, ``udp`` and ``rxm`` providers (and ``shm`` on Linux).
That is enough for one machine and for clusters on Ethernet. A wheel cannot
use a cluster's high-speed network (InfiniBand verbs, Slingshot, Omni-Path);
on such a cluster, install from source against the site's libfabric (or
libibverbs), as described below.

LCI uses the first provider libfabric offers. If a program does not start,
choose the TCP provider explicitly::

    $ export FI_PROVIDER=tcp

To check the installation::

    $ python -m charmrun.start +p2 hello_world.py

with ``hello_world.py`` from ``examples/simple`` in the Charm4py source.

Installing from source
----------------------

::

    $ git clone https://github.com/charmplusplus/charm4py.git
    $ cd charm4py
    $ pip install .

(``pip install charm4py`` on a platform without a wheel does the same from
the source distribution.)

When none of the ``CHARM4PY_CHARM_*`` variables below is set, the build
clones Charm++ at the commit named in ``charm_version.txt`` (with Reconverse,
the ``contrib/reconverse`` submodule) and builds it with::

    ./build charm4py <triplet> -j<N> --with-production --disable-fortran \
        --with-refnum-type="unsigned int" \
        --with-cmake-args=-DLCI_OPTIMIZE_FOR_NATIVE=OFF

before it builds the extension module, ``charm4py._ckdyn``. This takes several
minutes. ``<triplet>`` is the Reconverse triplet of the machine
(``reconverse-linux-x86_64``, ``reconverse-linux-arm8`` or
``reconverse-darwin-arm8``), and ``<N>`` is the number of cores. The clone and
the Charm++ build are kept in ``build/charm`` of the source tree, so a second
``pip install .`` reuses them. Requirements:

* git, a C and C++ compiler (C++17), and network access (the build fetches
  Charm++ and, through Reconverse, LCI);
* CMake 3.21 or newer (pip provides one when the system's is older);
* libfabric with its headers (``libfabric-dev`` on Debian and Ubuntu,
  ``libfabric-devel`` on RHEL-like systems, ``libfabric`` in Homebrew or
  MacPorts), or libibverbs (``rdma-core``). LCI looks for libfabric through
  pkg-config and in ``$OFI_ROOT``.

Environment variables that change this build:

* ``CHARM_BUILD_PROCESSES``: the ``-j`` count of the Charm++ build.
* ``CHARM_EXTRA_BUILD_OPTS``: more options for Charm++'s ``./build``, for
  example ``--with-cmake-args=-DLCI_NETWORK_BACKENDS=ofi`` to build LCI's
  libfabric backend only.
* ``CHARM4PY_CHARM_CACHE_DIR``: where to keep the clone and the build instead
  of ``build/charm``.
* ``CHARM4PY_BUILD_TRIPLET``: the Charm++ triplet, when the automatic choice
  is wrong. It must be a ``reconverse-*`` triplet.

Building against your own Charm++ build
---------------------------------------

For work on Charm++ or on Charm4py's extension module, point the build at a
Charm++ checkout or build:

* ``CHARM4PY_CHARM_BUILD_DIR``: an existing ``charm4py`` build of Charm++ (the
  directory with ``lib/libcharm.*`` and ``include/ckdyn.h``). It is used as it
  is; rebuild it yourself (``make`` in that directory) after changing Charm++.
* ``CHARM4PY_CHARM_DIR``: a Charm++ source tree. Without
  ``CHARM4PY_CHARM_BUILD_DIR``, the build uses
  ``$CHARM4PY_CHARM_DIR/<triplet>-charm4py`` when it holds a ``libcharm``,
  and otherwise builds it there with the command above.

The Charm++ tree needs CkDyn (``include/ckdyn.h``): the commit in
``charm_version.txt`` or a later one. Initialize Reconverse first::

    $ git -C /path/to/charm submodule update --init contrib/reconverse

To build that tree by hand::

    $ cd /path/to/charm
    $ ./build charm4py reconverse-darwin-arm8 -j8 --with-production \
          --disable-fortran --with-refnum-type="unsigned int" \
          --destination=$PWD/reconverse-darwin-arm8-charm4py

``--with-refnum-type="unsigned int"`` is required: Charm4py keeps future ids
in the 32-bit reference number of a message, and the extension module refuses
at import a ``libcharm`` built without it. ``--enable-tracing`` adds
Projections and Summary tracing; a ``libcharm`` built with it writes trace
files on every run, so the default build leaves it out (add it with
``CHARM_EXTRA_BUILD_OPTS=--enable-tracing``).
Use a build directory of its own for the ``charm4py``
target; a directory configured for the ``charm++`` target does not produce
the shared ``libcharm``.

An editable install for development::

    $ python -m venv venv && . venv/bin/activate
    $ pip install scikit-build-core nanobind numpy greenlet ninja
    $ CHARM4PY_CHARM_DIR=/path/to/charm \
      CHARM4PY_BUILD_TRIPLET=reconverse-darwin-arm8 \
      CHARM4PY_CHARM_BUILD_DIR=/path/to/charm/reconverse-darwin-arm8-charm4py \
      pip install --no-build-isolation -e .

With an editable install, Python changes take effect at once; after a change to
``src/*.cpp`` or to ``libcharm``, run the ``pip install`` command again.
On Linux the build needs ``patchelf`` (pip installs it in an isolated build;
with ``--no-build-isolation``, ``pip install patchelf`` first).

The test suite, from the source tree::

    $ CHARM4PY_TEST_NUM_PROCESSES=2 python auto_test.py

Running: one PE per process
---------------------------

Reconverse runs one Python interpreter per process, and Charm4py runs one PE
in each process for now. ``+pN`` with the Charm4py launcher starts ``N``
processes on this machine::

    $ python -m charmrun.start +p2 examples/simple/hello_world.py

The launcher bootstraps the processes the way LCI's ``lcrun`` does and passes
``+pe 2`` to Reconverse; when one process fails, it stops the others.
``lcrun`` itself is a script in LCI's source tree (``_deps/lci-src/lcrun`` in a
Charm++ build directory) and is not part of an installation; the equivalent
``lcrun`` command is::

    $ /path/to/lci/lcrun -n 2 python examples/simple/hello_world.py +pe 2

Across several nodes, start one process per PE with the cluster's launcher
(for example ``srun -n 8 python myprogram.py +pe 8``). ``+pe N`` given to a
single process (more PEs than processes) is refused at start-up. See
:doc:`running`.

GPUs (CUDA and HIP)
-------------------

Wheels have no device support. A CUDA or HIP build is a source build against
a Charm++ built with ``cuda`` or ``hip`` and the site's CUDA, libfabric and
PMI. The recipe (``CHARM_EXTRA_BUILD_OPTS=cuda``) is written and checked in
step 3 of the Charm4py 2 plan (``docs/design/DESIGN.md``, section 6.4); until
then this section is a placeholder.

Known issues
------------

* The pin in ``charm_version.txt`` is a commit of Charm++'s ``ckdyn-dev``
  branch; it moves to a ``recharm`` commit when CkDyn is merged there.
* No free-threaded (``3.13t``) wheels yet, and no multi-PE processes (step 7
  of the plan).

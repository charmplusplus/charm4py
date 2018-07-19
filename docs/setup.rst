============
Setup
============

.. .. contents::

Before CharmPy setup, please download and build the Charm++ shared library as explained
in the previous section. Next, enter the CharmPy source code directory and run::

    $ python setup.py CHARM_DIR

where ``CHARM_DIR`` is the root folder of your Charm++ installation.
A configuration file (``charmpy.cfg``) will be automatically created after setup.

Setup will look for ``cython`` or ``cffi`` and select the best mode. If one of those is
found, it will compile a module to access the shared library and set it as
the default interface in ``charmpy.cfg``. Make sure that you
use the same compiler to build both the Charm++ library and the interface module.
If there are issues compiling the interface module, please refer to the "Troubleshooting"
section below.

Cleanup
-------

After setup, you only need ``libcharm.so`` and ``charmrun`` executable, which are
located in ``CHARM_DIR/lib`` and ``CHARM_DIR/bin`` respectively. You can move these
files to directories of your choice and remove the Charm++ installation if desired.

Troubleshooting
---------------

Building CFFI library wrapper module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- On macOS, with Python 2.7 additional flags might be necessary during setup to avoid
  compilation problems like "unknown type name '__int128_t'"::

  $ ARCHFLAGS="-arch x86_64" python setup.py CHARM_DIR

- If a valid version of CFFI is installed but issues persist in compiling or using
  the CFFI library wrapper, the ctypes interface can be forced by doing::

    $ python setup.py CHARM_DIR --with-ctypes

  (note that ctypes has reduced performance).

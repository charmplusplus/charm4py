
CharmPy
=======

This module allows writing Charm++ applications in Python. All of the application
code and entry methods can be written entirely in Python. The core Charm++ runtime
is implemented in a C/C++ shared library which the charmpy module interfaces with.

As with any Python program, there are several methods available to support
high-performance functions. These include, among others: 'numpy'; writing the
desired functions in Python and JIT compiling to native machine instructions using
'numba'; or accessing C or Fortran code using 'f2py'. Another option for increased
speed is to run the program using a fast Python implementation (e.g. PyPy).

Example applications are in the 'examples' subdirectory.


Installing
==========

Requirements:

  - Python 2.7 or higher. Python 3 recommended for best performance and for
    continued support.
    CharmPy has been tested with the following implementations: CPython (most common
    Python implementation) and PyPy.

  - Charm++ shared library. See 'README.charmpy' in Charm++ distribution (branch
    'jjgalvez/charmpy') for instructions on building the shared library. NOTE: this
    is required prior to setup of charmpy.

  - There are two modes to access the shared library from CharmPy: cffi and ctypes.
    For best performance, cffi is highly recommended (see below for instructions).

Using CharmPy with CFFI
-----------------------
CFFI version >= 1.7 required
If your system has an older version installed, remove that one first.

To install a recent version, please see:
https://cffi.readthedocs.io/en/latest/installation.html
Typically, 'pip install cffi' will suffice.

NOTE: If you have multiple versions of Python installed (e.g. python2, python3,
pypy), make sure cffi is installed for the version that you will use to run charmpy.
For example, if you are installing cffi with pip, make sure you use the version of
pip that corresponds to the interpreter you'll be using.

In addition, development files for your Python distribution will likely be needed
(e.g. Python.h) to compile the charm library wrapper. On Ubuntu Linux, these are
usually provided by 'python3-dev' or similar package.

Setup
-----

A one-time setup is needed before running CharmPy programs, and the Charm++ shared
library (libcharm.so) must be built prior to this, and located in CHARM_DIR/lib

Next, do:
> python setup.py CHARM_DIR
where CHARM_DIR is the directory where Charm++ is located.

NOTE: If using cffi, setup is needed for every Python implementation that will run
CharmPy.

A configuration file (charmpy.cfg) will be automatically created and placed in the
same directory as charmpy.py. If a valid version of cffi is installed, setup will
compile the library wrapper and set cffi as default interface in 'charmpy.cfg'.
If there are issues compiling the library wrapper, please see "Troubleshooting"
section at the end.

NOTE: A charmpy.cfg file in user's home directory will override the default
charmpy.cfg.


Running CharmPy programs
========================

Allow CharmPy to find the Charm++ shared library
------------------------------------------------

* Using ctypes: the path to libcharm.so is set during setup in charmpy.cfg, and no
other steps are necessary. This setting can be overridden by modifying charmpy.cfg
or by setting the environment variable 'LIBCHARM_PATH':

  > export LIBCHARM_PATH=/path/to/libcharm_folder

* Using cffi: libcharm.so must reside in a directory where the dynamic linker will
find it. The method to achieve this varies by system:

  On Linux, one way to accomplish this is:
  > export LD_LIBRARY_PATH=/path/to/libcharm_folder

  On Mac, DYLD_LIBRARY_PATH might work, or else library can be placed or
  symlinked to a system library directory, like /usr/local/lib
  More info related to Mac is in troubleshooting section at the end.

Launching CharmPy programs
--------------------------

charmpy module must be in Python path:
> export PYTHONPATH=path_to_charmpy_folder (the folder that contains charmpy.py)

To launch CharmPy programs, use the charmrun binary generated when building the
Charm++ library. For example:

./charmrun +p4 /usr/bin/python examples/wave2d/wave2d.py ++local

Error output during startup
---------------------------

Output during startup (before Charm++ makes the call to register the main module) is
suppressed by charmrun. After the call to register the main module, CharmPy
redirects the output to the descriptors opened by Charm.

In other words, errors during startup, including Python syntax errors, will not be
printed. To see them, launch the program without charmrun, e.g.:
/usr/bin/python examples/wave2d/wave2d.py

Shared-memory parallelism
-------------------------

Python multithreading is not recommended as the GIL in most Python implementations
prevents threads from running concurrently. Because Python entry methods cannot run
concurrently for a given process, Charm++ should be built in non-SMP mode.

However, shared-memory parallel code can be run inside an entry method. Some options
include:
- numba with 'parallel' flag [1]
- f2py and OpenMP code
- Python multiprocessing library (using Shared memory)

If such options are used, less number of CharmPy processes than cores would
typically be launched.


Troubleshooting
===============

Issues with cffi library wrapper
--------------------------------

- Solutions for some issues observed in macOS environment:

  1. For loading libraries through dynamic linker without turning off SIP, the
     following symlink solution can be used:
     $ ln -s /path/to/libcharm.so /usr/local/lib

  2. With Python 2.7 additional flags might be necessary during setup to avoid
     compilation problems like "unknown type name '__int128_t'":
     $ ARCHFLAGS="-arch x86_64" python setup.py CHARM_DIR

  3. While executing a CharmPy program with cffi interface if an error related
     to unsafe use of relative rpath libcharm.so appears, try the following:
     $ cd charmpy/__cffi_objs__
     $ install_name_tool -change libcharm.so /usr/local/lib/libcharm.so _charmlib.so

     Note : This assumes you've symlinked the library as indicated in (1).

- If a valid version of cffi is installed but issues persist in compiling or using
the cffi library wrapper, the ctypes interface can be forced by doing:

  > python setup.py CHARM_DIR --with-ctypes

  Note that ctypes has reduced performance.

References
==========

[1] http://numba.pydata.org/numba-doc/dev/user/parallel.html


AUTHORS
=======

Juan Galvez
Karthik Senthil

Original framework and module by:
Juan Galvez <jjgalvez@illinois.edu>

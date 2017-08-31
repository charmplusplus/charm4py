
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

The existing ctypes version of CharmPy does not require compilation and is
compatible with any Python interpreter that supports ctypes.

Example applications are in the 'examples' subdirectory.

Running CharmPy programs
========================

Build Charm++ shared library
----------------------------

You need to build Charm++ as a shared library. The branch that currently has support
for CharmPy is 'juan/charmpy'. Instructions to build the library are in that branch,
in README.charmpy

Launching CharmPy programs
--------------------------

Allow Python/ctypes to find the shared library. On Linux, one method is:
> export LD_LIBRARY_PATH=path_to_shared_lib_folder

charmpy module must be in Python path:
> export PYTHONPATH=path_to_charmpy_folder

To launch CharmPy programs, use the charmrun binary generated when building the
Charm++ library. For example:

./charmrun +p4 /usr/bin/python examples/wave2d/wave2d.py ++local

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

References
==========

[1] http://numba.pydata.org/numba-doc/dev/user/parallel.html


AUTHORS
=======

Original framework and module by:
Juan Galvez <jjgalvez@illinois.edu>

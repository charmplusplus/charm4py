============
Install
============

.. .. contents::

Charm4py runs on Linux, macOS, Windows, Raspberry Pi, and a wide variety of clusters and
supercomputer environments (including many supercomputers in the TOP500).

Charm4py runs on Python 3.7+. Charm4py has been tested with the
following Python implementations:
CPython (most common implementation) and PyPy_.


.. _PyPy: https://pypy.org

Installing Charm4Py binaries (via pip)
---------------------------------------

The easiest way to install Charm4Py is via pip. Currently, pip wheels are available for Linux and macOS.

To install the latest release of Charm4Py, run::

    $ pip install charm4py

This will install the latest stable release of Charm4Py, using the default underlying Charm++ build
(see the `Charm++ manual`_ for more information on the different builds of Charm++). If you want to 
use a specific Charm++ build, you can install and build Charm4Py from source. Note that the source distribution
is available via "pip install", but the standard from source build process is via "git clone", as outlined below.

Charm4Py provides a small number of examples with the binary distribution that can be used as a sanity check to verify basic functionality of the installation.
Examples can be run via the following command line tool installed with Charm4Py::

    $ charm4py_test [example]

Currently, Charm4Py binaries are distributed with the following examples:

- 'group_hello' - A simple hello world example wherein all members of a group print a message.
- 'array_hello' - A simple hello world example wherein all elements of an array print a message one at a time, passing a message to the next element.
- 'simple_ray' - A simple example of a Ray application.

To run a comprehensive set of tests, install Charm4py from source.

Installing Charm4Py from source
------------------------------------------------------------

This install process covers the installation of Charm4Py from source.

The first step is to clone the Charm4py repository from Git::

    $ git clone https://github.com/charmplusplus/charm4py.git
    $ cd charm4py

Next, clone the Charm++ repo into charm_src::

    $ git clone https://github.com/charmplusplus/charm.git charm_src/charm

Once this is done, there are two ways to build Charm4py. The first is to simply run the installation
from the Charm4py root. This method will use the default Charm++ backend::

    $ cd ..
    $ pip install .

The other option is to manually build Charm++ before building Charm4py. This may be necessary
if you want to configure Charm++ differently from the default. To do this, change to
the charm directory and run the following build command, then build Charm4Py::

    $ cd charm
    $ ./build charm4py <target-architecture> -j<N> --with-production
    $ cd ../..
    $ pip install .

Finally, if necessary, when installing dependencies or when running the install script, add the --user
option to the Python command to complete the installation without permission errors.

After building, you can run Charm4py examples. One example you can try is
array_hello.py, which can be run as follows::

    $ cd examples/hello
    $ python -m charmrun.start +p2 array_hello.py


To run the full Charm4py test suite after building from source, run the following command from the repository root::

    $ python auto_test.py

Choosing your target architecture
------------------------------------------------------------

When building from source, as described above, you must chose the appropriate target architecture.

For building on a laptop or personal machine, you must use a netlrts build of Charm4Py. 
For example, to build for a personal machine running macOS with ARM processors, using 4 cmake 
threads, you would run::
    
    $ ./build charm4py netlrts-darwin-arm -j4 --with-production

To install Charm4Py on a cluster machine, you will generally want to chose a different backend. 
For example, to use Charm4Py with MPI, build the Charm backend as follows::

    $ ./build charm4py mpi-<os>-<architecture> -j<N> --with-production

Check the Charm++ documentation to identify the correct os and architecture command 
to pass into the build command. 

.. _manual: https://charm.readthedocs.io/en/latest/charm++/manual.html#installing-charm

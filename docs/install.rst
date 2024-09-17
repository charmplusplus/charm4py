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

To minimally test the pip installation, you can run the following one-line example::

    $ python -c 'import charm4py; charm4py.charm.start(lambda x: print("hello") or charm4py.charm.exit())'

To run the full test suite, it is necessary to build Charm4Py from source. Examples of charm4py programs are also available in the Charm4Py source.

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

After building, you can run Charm4py examples. For example::

    $ python -m charmrun.start +p2 examples/hello/array_hello.py

The full charm4py test suite can be run from the Charm4Py root as follows::

    $ python auto_test.py

Choosing the target architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

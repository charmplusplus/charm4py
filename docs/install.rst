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


Installing Charm4Py on a laptop/personal machine
------------------------------------------------

This install process covers the installation of Charm4Py on a laptop or personal machine, as opposed to a cluster.

Before installing, you need the following prerequisites:
    - CPython: numpy, greenlet and cython (``pip3 install 'numpy>=1.10.0' cython greenlet``)
    - PyPy: none

You can get these prerequisites by running the following command::

    $ pip3 install -r requirements.txt

The first step is to clone the Charm4py repository from Git::

    $ git clone https://github.com/charmplusplus/charm4py.git
    $ cd charm4py

Next, create a folder called charm_src in the charm4py repo, and then clone the Charm++ repo
into that folder::

    $ mkdir charm_src && cd charm_src
    $ git clone https://github.com/charmplusplus/charm.git

Once this is done, there are two ways to build Charm4py. The first way is to change back up
into the Charm4Py directory and run the install script::
    
    $ cd ..
    $ python3 setup.py install

The other option is to manually build Charm++ before building Charm4py. To do this, change to
the charm directory and run the following build command::
    
    $ cd charm
    $ ./build charm4py netlrts-<os>-<architecture> -j<N> --with-production

For building on a laptop, you must use a netlrts build of Charm4Py. Check the Charm++ documentation
to identify the correct os and architecture command to pass into the build command. The -j option
is a cmake option that launches N threads for the make.

Then, return to the charm4py directory and run setup.py::

    $ cd ../..
    $ python3 setup.py install


After building, you can run Charm4py examples. One example you can try is 
array_hello.py, which can be run as follows::

    $ cd examples/hello
    $ python -m charmrun.start +p2 array_hello.py

Installing Charm4Py on a cluster machine
----------------------------------------

To install Charm4Py on a cluster machine, you will generally follow the same steps as above, but
with the following changes. First, when building Charm++, use the MPI build instead of the netlrts
build::

    $ ./build charm4py mpi-<os>-<architecture> -j<N> --with-production

Next, pass in the MPI option to the python setup script::

    $ python3 setup.py install --mpi

Finally, if necessary, when installing dependencies or when running the install script, add the --user
option to the Python command to complete the installation without permission errors.



.. _manual: https://charm.readthedocs.io/en/latest/charm++/manual.html#installing-charm

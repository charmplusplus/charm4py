============
Running
============

.. .. contents::

CharmPy includes a launcher called ``charmrun`` to run parallel applications on
desktops and small clusters. Supercomputers and some clusters provide their
own application launchers (these can also be used to launch CharmPy applications).

charmrun
--------

After installing CharmPy as explained in the previous section, you can launch
applications like this::

    $ python -m charmrun.start +p4 myprogram.py

The option ``+pN`` specifies how many processes to run the application with.

Alternatively, if ``charmrun`` is in your PATH (this depends on where charmpy was
installed and your system configuration)::

    $ charmrun +p4 myprogram.py

CharmPy programs accept the `same command-line parameters`_ as Charm++.

.. _same command-line parameters: http://charm.cs.illinois.edu/manuals/html/charm++/C.html


Running on Multiple Hosts
~~~~~~~~~~~~~~~~~~~~~~~~~

``charmrun`` can run an application on multiple hosts (e.g. a network of workstations)
by passing it a file containing the list of nodes (*nodelist* file). Hosts can be
specified by IP address or host name. For example, this is a simple nodelist file
specifying four hosts::

    group mynodes
        host 192.168.0.10
        host 192.168.0.133
        host myhost
        host myhost2

The application can be launched like this::

    $ charmrun +pN myprogram.py ++nodelist mynodelist.txt

With this method, charmrun uses ``ssh`` to log into remote machines and spawn processes.

charmrun can also use the cluster's ``mpiexec`` job launcher instead of the built in ssh method.

See the `charmrun manual`_ for more information and alternative ways to work with nodelist
files.

.. _charmrun manual: http://charm.cs.illinois.edu/manuals/html/charm++/C.html


Troubleshooting
~~~~~~~~~~~~~~~

Issue
    Program hangs with no output when launching with ``charmrun``.

Solution
    This typically occurs when launching the program on multiple hosts, and an error
    ocurring before starting charm (e.g. syntax error). To diagnose, launch the
    program on a single host.


Using system job launchers
--------------------------

CharmPy applications can also be launched using system job launchers
(e.g. aprun, ibrun, SLURM).
The exact details of how to do so depend on the system, and typically Charm++ has
to be built with a specialized network layer like MPI, GNI or OFI
(see `Charm++ manual build`__).

.. __: install.html#manually-building-the-charm-shared-library

In all cases, the mechanism consists in launching one or multiple Python processes
on each node, and passing the main application file to Python. Here is a simple script
for SLURM on a Cray-based system:

.. code-block:: bash

    #!/bin/bash -l
    #SBATCH -N 8         # number of nodes
    #SBATCH -t 00:30:00
    #SBATCH -C knl

    module load craype-hugepages8M
    module load python/3.6-anaconda-4.4

    export PYTHONPATH=/path/to/charmpy
    PYTHON_EXEC=`which python3`

    srun -n 512 -c 1 $PYTHON_EXEC myprogram.py app_param1 app_param2 ...

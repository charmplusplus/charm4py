============
Running
============

.. .. contents::

Charm4py includes a launcher called ``charmrun`` to run parallel applications on
desktops and small clusters. Supercomputers and some clusters provide their
own application launchers (these can also be used to launch Charm4py applications).

charmrun
--------

After installing Charm4py as explained in the previous section, you can launch
applications like this::

    $ python3 -m charmrun.start +p4 myprogram.py

The option ``+pN`` specifies how many processes to run the application with.

.. note::
  Charm4py on Reconverse runs one PE per process for now. Use several
  processes with one PE each (``+pN`` with the launcher above, or
  ``lcrun -n N python myprogram.py +pe N``). Giving ``+pe N`` to a single
  process stops the program at start-up with an error saying so.

Alternatively, if ``charmrun`` is in your PATH (this depends on where Charm4py was
installed and your system configuration)::

    $ charmrun +p4 myprogram.py

You can launch an *interactive shell* using the ``++interactive`` option, for
example::

    $ python3 -m charmrun.start +p4 ++interactive

Charm4py programs accept the `same command-line parameters`_ as Charm++.

.. _same command-line parameters: https://charm.readthedocs.io/en/latest/charm++/manual.html#running-charm-programs



Running on multiple hosts
~~~~~~~~~~~~~~~~~~~~~~~~~

The Charm4py launcher starts processes on the local machine only: the
``++nodelist`` file and the ssh start-up of the 1.x ``charmrun`` binary are
part of classic Converse, which Charm4py 2 does not use. On several hosts,
start the processes with the cluster's launcher (`Using system job
launchers`_) or with LCI's ``lcrun``.


Using charmrun from a Python program
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can launch a Charm4py application from inside a Python application,
and wait for it to complete, in this manner:

.. code-block:: python

    from charmrun import start

    start.start(['+p4', 'myprogram.py'])  # launch parallel application and wait for completion


Note that you can also use Python's ``subprocess`` library and launch the same command
as you would via the command line.


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

Charm4py applications can also be launched with system job launchers (srun,
mpiexec, aprun and others): start one Python process per PE, pass the main
application file to Python, and give the total PE count with ``+pe``. The
processes find each other through LCI's PMI support (PMI-1, PMI-2 or PMIx, as
the launcher provides). For SLURM:

.. code-block:: bash

    #!/bin/bash -l
    #SBATCH -N 8              # number of nodes
    #SBATCH --ntasks-per-node 16
    #SBATCH -t 00:30:00

    PYTHON_EXEC=`which python3`
    srun $PYTHON_EXEC myprogram.py +pe 128 app_param1 app_param2 ...

A wheel's libfabric has only its TCP-based providers; to use a cluster's
high-speed network, install Charm4py from source against the site's libfabric
(see :doc:`install`).

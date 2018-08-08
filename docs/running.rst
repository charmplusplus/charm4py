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

charmrun can also be used to launch a program on `a network of computers`_.

.. _a network of computers: http://charm.cs.illinois.edu/manuals/html/charm++/C.html#SECTION05330000000000000000

.. _same command-line parameters: http://charm.cs.illinois.edu/manuals/html/charm++/C.html


Troubleshooting
~~~~~~~~~~~~~~~

Issue
    Program hangs with no output when launching with ``charmrun``.

Solution
    This typically occurs when launching the program on multiple hosts, and an error
    ocurring before starting charm (e.g. syntax error). To diagnose, launch the
    program on a single host.

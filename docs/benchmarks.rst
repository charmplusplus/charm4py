============
Benchmarks
============

.. contents::

This section presents Charmpy benchmark results using: (a) real examples and miniapps;
(b) synthetic test cases to evaluate specific features.

Mini-app benchmarks
-------------------

LeanMD - Molecular Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO

Features benchmarks
-------------------

Bypass pickling for NumPy and other arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This feature is also known as "direct-copy". As explained in :ref:`perf-serialization-label`,
it is used when NumPy arrays and other structures supporting the buffer protocol are passed
as arguments of remote methods.

To test and evaluate the performance of this feature we wrote a small program
(``tests/test_dcopy.py``) where a chare array is created, and each element
sends three large data arrays to the rest of the elements, for a fixed number of iterations.
The experiment was carried out using 4 cores on a standard Macbook Pro. The results
are shown below (for 10 iterations):

+--------------------------+-----------------+--------------+----------------+
|         Metric           |  Without dcopy  |  With dcopy  | % speedup      |
+==========================+=================+==============+================+
|  Send time (s)           |      2.406      |    1.046     |     56.53%     |
+--------------------------+-----------------+--------------+----------------+
|  Receive time (s)        |      0.372      |    0.343     |      7.80%     |
+--------------------------+-----------------+--------------+----------------+
|  Total program time (s)  |     12.72804    |   11.21846   |     11.86%     |
+--------------------------+-----------------+--------------+----------------+
|  Bytes sent (MB)         |     1339.892    |   1339.323   |      0.04%     |
+--------------------------+-----------------+--------------+----------------+

Note: this feature is enabled by default with Python 3 and CFFI.

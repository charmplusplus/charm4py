=============
Serialization
=============

Usually a remote method invocation results in serialization of the arguments
into a message that is sent to a remote process.
For many situations, Charm4py relies on Python's ``pickle`` module.

.. important::
    Pickling is bypassed for certain data types that implement the buffer protocol
    (`byte arrays`_, array.array_ and `NumPy arrays`_) and is encouraged for best
    performance.
    For these, the data is directly copied from its memory buffer in Python into a message in the Charm
    C++ library for sending.
    The :doc:`perf-tips` section explains how to take advantage of this.


Pickling can account for much of the overhead of the Charm4py runtime. Fastest
pickling is obtained with the C implementation of the ``pickle`` module
(only available in CPython).
A general guideline to achieve good pickle performance is to avoid passing custom types as
arguments to remote methods in *the application's critical path*.
Examples of recommended types to use for best performance include: Python containers
(lists, dicts, set), basic datatypes (int, float, str, bytes) or combinations of the
above (e.g.: dict containing lists of ints). Custom objects are automatically
pickled but can significantly affect the performance of pickle and therefore their
use inside the critical path is not recommended.


.. _byte arrays: https://docs.python.org/3/library/stdtypes.html#bytes

.. _array.array: https://docs.python.org/3/library/array.html

.. _NumPy arrays: https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html

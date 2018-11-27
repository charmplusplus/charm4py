
readonlies
----------

``charm4py.readonlies`` is an object that serves as a container for data that
the application wants to broadcast to every process after the "entry point" has executed.
Attributes added to ``readonlies`` during execution of the entry point will become
available in the ``readonlies`` instance of every process.

For example:

.. code-block:: python

    from charm4py import charm, Chare, Group
    from charm4py import readonlies as ro

    class Test(Chare):
        def __init__(self):
            # this will print 3 and 5 on every PE
            print(ro.x, ro.y)

    def main(args):
        ro.x = 3
        ro.y = 5
        Group(Test)

    charm.start(main)

.. warning::

    Names of attributes added to ``readonlies`` must not start or end with '_'.

    As the name implies, data in ``readonlies`` is intended to be read-only.
    This is because the broadcast is done
    only once, and after this there is no synchronization of the data.

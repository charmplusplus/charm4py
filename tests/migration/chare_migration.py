
"""
A program to test migration of chares.
"""

from charmpy import charm, Mainchare, Chare, Array, CkMyPe, CkNumPes, CkExit, CkAbort


class Main(Mainchare):
    """
    Main chare.
    """

    def __init__(self, args):
        if CkNumPes() != 4:
            CkAbort("Run program with only 4 PEs")
        array_proxy = Array(Migrate, 4)
        array_proxy.start()


class Migrate(Chare):
    """
    A class to test the migration of chares.
    """

    def test(self):
        """
        Test method called after migration to assert that the
        chare has migrated.
        """
        print("Test called on PE ", CkMyPe())
        assert CkMyPe() == 1
        CkExit()

    def start(self):
        """
        Invoke the starter code for test.
        """
        if CkMyPe() == 0:
            print("On PE", CkMyPe(), "before migration")
            self.thisProxy[self.thisIndex].test()
            self.migrate(1)


# ---- start charm ----
charm.start()

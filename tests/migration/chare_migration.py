
"""
A program to test migration of chares.
"""

from charmpy import charm, Mainchare, Chare, Array, CkMyPe


CHARES_PER_PE = 1

class Main(Mainchare):
    """
    Main chare.
    """

    def __init__(self, args):
        if charm.numPes() == 1:
            charm.abort("Run program with more than 1 PE")
        array_proxy = Array(Migrate, CHARES_PER_PE * charm.numPes())
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
        if self.thisIndex == (0,):
            print("Test called on PE ", CkMyPe())
        assert CkMyPe() == self.toPe
        self.contribute(None, None, self.thisProxy[0].done)

    def done(self):
        charm.exit()

    def start(self):
        """
        Invoke the starter code for test.
        """
        if CkMyPe() == 0:
            print("On PE", CkMyPe(), "before migration")
        self.thisProxy[self.thisIndex].test()
        self.toPe = (charm.myPe() + 1) % charm.numPes()
        self.migrate(self.toPe)


# ---- start charm ----
charm.start()

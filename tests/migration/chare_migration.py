
"""
A program to test migration of chares.
"""

from charm4py import charm, Chare, Array


CHARES_PER_PE = 1


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
            print("Test called on PE ", charm.myPe())
        assert charm.myPe() == self.toPe
        self.contribute(None, None, charm.thisProxy[0].exit)

    def start(self):
        """
        Invoke the starter code for test.
        """
        if charm.myPe() == 0:
            print("On PE", charm.myPe(), "before migration")
        self.thisProxy[self.thisIndex].test()
        self.toPe = (charm.myPe() + 1) % charm.numPes()
        self.migrate(self.toPe)


def main(args):
    if charm.numPes() == 1:
        charm.abort("Run program with more than 1 PE")
    array_proxy = Array(Migrate, CHARES_PER_PE * charm.numPes())
    array_proxy.start()


charm.start(main)


"""
A program to test migration of chares.
"""

from charm4py import charm, Chare, Array, coro


CHARES_PER_PE = 1


class Migrate(Chare):
    """
    A class to test the migration of chares.
    """

    @coro
    def start(self):
        """
        Invoke the starter code for test.
        """
        print(self.thisIndex, 'on PE', charm.myPe(), 'before migration')
        self.toPe = (charm.myPe() + 1) % charm.numPes()
        self.migrateAndWait(self.toPe)
        print(self.thisIndex, 'migrated to PE', charm.myPe())
        assert charm.myPe() == self.toPe
        self.contribute(None, None, charm.thisProxy[0].exit)



def main(args):
    if charm.numPes() == 1:
        charm.abort('Run program with more than 1 PE')
    array_proxy = Array(Migrate, CHARES_PER_PE * charm.numPes())
    array_proxy.start()


charm.start(main)

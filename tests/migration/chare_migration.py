
"""
A program to test migration of chares.
"""

from charm4py import charm, Chare, Array, Future


CHARES_PER_PE = 1


class Migrate(Chare):
    """
    A class to test the migration of chares.
    """

    def migrated(self):
        """
        Test method called after migration to assert that the
        chare has migrated.
        """
        print(self.thisIndex, 'migrated to PE', charm.myPe())
        assert charm.myPe() == self.toPe

    def start(self):
        """
        Invoke the starter code for test.
        """
        if charm.myPe() == 0:
            print(self.thisIndex, 'on PE', charm.myPe(), 'before migration')
        self.toPe = (charm.myPe() + 1) % charm.numPes()
        self.thisProxy[self.thisIndex].migrate(self.toPe)

class OtherMigrate(Chare):
    def print(self):
        print(self.thisIndex)


def main(args):
    if charm.numPes() == 1:
        charm.abort('Run program with more than 1 PE')
    array_proxy = Array(Migrate, CHARES_PER_PE * charm.numPes())
    bound_array = Array(OtherMigrate, CHARES_PER_PE * charm.numPes(), boundTo=array_proxy)
    #array_proxy.prox(bound_array)
    #array_proxy.start()
    array_proxy.start()


charm.start(main)

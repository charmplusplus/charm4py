from charm4py import charm, Chare, Array
import sys
sys.argv += ['+balancer', 'RandCentLB']

MAX_ITER = 100


class Test(Chare):

    def __init__(self):
        self.iteration = 0
        self.originalPe = charm.myPe()
        if self.thisIndex[0] % 2 == 0:
            self.setMigratable(False)

    def start(self):
        self.hasMigrated = False
        self.iteration += 1
        if self.iteration == MAX_ITER:
            self.contribute(None, None, charm.thisProxy[0].exit)
        else:
            self.prevPe = charm.myPe()
            self.AtSync()

    def resumeFromSync(self):
        assert self.migratable == (self.thisIndex[0] % 2 != 0)
        if not self.migratable:
            assert charm.myPe() == self.originalPe
        assert self.hasMigrated == (self.prevPe != charm.myPe())
        self.thisProxy[self.thisIndex].start()

    def migrated(self):
        self.hasMigrated = True


def main(args):
    array = Array(Test, charm.numPes() * 8, useAtSync=True)
    array.start()


charm.start(main)

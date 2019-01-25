from charm4py import charm


def main(args):
    pe = charm.firstPeOnPhysicalNode(0)
    assert pe in list(range(charm.numPes()))

    phyNodes = set([])
    for pe in range(charm.numPes()):
        phyNodes.add(charm.physicalNodeID(pe))
    assert len(phyNodes) == charm.numPhysicalNodes()

    exit()


charm.start(main)

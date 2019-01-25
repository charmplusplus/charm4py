from charm4py import charm


def main(args):

    # phynode IDs should go from 0 to N-1, but assume that is not the case
    phyNodes = set()
    for pe in range(charm.numPes()):
        phynode = charm.physicalNodeID(pe)
        assert phynode >= 0
        phyNodes.add(phynode)

    assert len(phyNodes) == charm.numPhysicalNodes()

    for phynode in phyNodes:
        pe = charm.firstPeOnPhysicalNode(phynode)
        assert pe >= 0 and pe < charm.numPes()
        assert charm.physicalNodeID(pe) == phynode

    exit()


charm.start(main)

from charm4py import charm


def main(args):
    charm.firstPeOnPhysicalNode(0)

    phyNodes = set([])
    for pe in range(charm.numPes()):
        phyNodes.add(charm.physicalNodeID(pe))
    assert len(phyNodes) == charm.numPhysicalNodes()

    exit()


charm.start(main)

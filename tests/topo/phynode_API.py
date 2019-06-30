from charm4py import charm


def main(args):
    myhost = charm.myHost()
    assert myhost >= 0 and myhost < charm.numHosts()
    allpes = set()
    rank0pes = set()
    totalpes = 0
    for host in range(charm.numHosts()):
        pes = charm.getHostPes(host)
        assert charm.getHostNumPes(host) == len(pes)
        assert len(allpes.intersection(pes)) == 0
        allpes.update(pes)
        for i, pe in enumerate(pes):
            assert charm.getPeHost(pe) == host
            assert charm.getPeHostRank(pe) == i
        rank0 = charm.getHostFirstPe(host)
        assert rank0 == pes[0]
        assert rank0 >= 0 and rank0 < charm.numPes()
        assert rank0 not in rank0pes
        rank0pes.add(rank0)
        totalpes += charm.getHostNumPes(host)
    assert sorted(allpes) == list(range(charm.numPes()))
    assert totalpes == charm.numPes()
    exit()


charm.start(main)

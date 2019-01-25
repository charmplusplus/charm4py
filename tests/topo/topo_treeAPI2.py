from charm4py import charm


def explore_tree(root, pes, found_pes):

    subtrees = charm.getTopoSubtrees(root, pes, bfactor=2)
    for subtree in subtrees:
        # make sure list of PEs doesn't have duplicates
        assert len(set(subtree)) == len(subtree)
        # make sure these PEs didn't appear in any other subtree
        assert len(found_pes.intersection(set(subtree))) == 0
        found_pes.add(subtree[0])
        explore_tree(subtree[0], subtree, found_pes)


def main(args):

    pes = list(range(charm.numPes()))
    found_pes = set([0])
    explore_tree(0, pes, found_pes)
    assert sorted(list(found_pes)) == pes
    exit()


charm.start(main)

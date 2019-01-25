from charm4py import charm

allPes_check  = []
evenPes_check = []

def printWholeTree(root, current):
    allPes_check.append(current)
    parent, children = charm.getTopoTreeEdges(current, root, bfactor=2)
    print(current, "children=", children)
    for c in children: printWholeTree(root, c)


def printEvenNbTree(pes, current):
    evenPes_check.append(current)
    parent, children = charm.getTopoTreeEdges(current, pes[0], pes, bfactor=2)
    print(current, "children=", children)
    for c in children: printEvenNbTree(pes, c)


def main(args):

    global allPes_check, evenPes_check

    print("\nWhole topo tree rooted at PE 0")
    printWholeTree(0, 0)
    assert(len(allPes_check) == charm.numPes() and set(allPes_check) == set(range(charm.numPes())))
    allPes_check = []

    lastPE = charm.numPes() - 1
    if lastPE != 0:
        print("\nWhole topo tree rooted at", lastPE)
        printWholeTree(lastPE, lastPE)
        assert(len(allPes_check) == charm.numPes() and set(allPes_check) == set(range(charm.numPes())))
        allPes_check = []

    print("\nEven numbered PE tree, rooted at PE 0")
    evenPEs = [pe for pe in range(charm.numPes()) if pe % 2 == 0]
    printEvenNbTree(evenPEs, 0)
    assert(len(evenPes_check) == len(evenPEs) and set(evenPes_check) == set(evenPEs))
    evenPes_check = []

    newRoot = evenPEs[-1]
    if newRoot != 0:
        evenPEs.insert(0, evenPEs.pop())  # move root from back to beginning of list
        print("\nEven numbered PE tree, rooted at PE", newRoot)
        printEvenNbTree(evenPEs, newRoot)
        assert(len(evenPes_check) == len(evenPEs) and set(evenPes_check) == set(evenPEs))
        evenPes_check = []

    exit()


charm.start(main)

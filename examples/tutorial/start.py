from charm4py import charm

def main(args):
    print("Charm program started on processor", charm.myPe())
    print("Running on", charm.numPes(), "processors")
    exit()

if __name__ == '__main__':
    charm.start(main)

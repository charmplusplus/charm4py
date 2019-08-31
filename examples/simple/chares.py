from charm4py import charm, Chare, Group, Array


class MyChare(Chare):

    def __init__(self):
        print('Hello from MyChare instance in processor', charm.myPe())


def main(args):
    # Note that chare creation calls are also asynchronous

    # create one instance of MyChare on every processor
    my_group = Group(MyChare)

    # create 3 instances of MyChare, distributed among cores by the runtime
    my_array = Array(MyChare, 3)

    # create 2 x 2 instances of MyChare, indexed using 2D index and distributed
    # among cores by the runtime
    my_2d_array = Array(MyChare, (2, 2))

    # wait for the chare collections to be created
    charm.awaitCreation(my_group, my_array, my_2d_array)
    exit()


charm.start(main)

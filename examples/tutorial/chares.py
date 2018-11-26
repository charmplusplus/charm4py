from charm4py import charm, Chare, Group, Array

class MyChare(Chare):
    def __init__(self):
        print("Hello from MyChare instance in processor", charm.myPe())

    def work(self, data):
      pass

def main(args):

    # create one instance of MyChare on every processor
    my_group = Group(MyChare)

    # create 3 instances of MyChare, distributed among all cores by the runtime
    my_array = Array(MyChare, 3)

    # create 2 x 2 instances of MyChare, indexed using 2D index and distributed
    # among all cores by the runtime
    my_2d_array = Array(MyChare, (2, 2))

    charm.awaitCreation(my_group, my_array, my_2d_array)
    exit()

if __name__ == '__main__':
    charm.start(main)

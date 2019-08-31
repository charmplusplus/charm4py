from charm4py import charm, Chare, Group, Reducer


class MyChare(Chare):

    def work(self, data):
        # members of the group do a 'sum' reduction, result will be sent
        # to method 'collectResult' of element 0
        self.reduce(self.thisProxy[0].collectResult, data, Reducer.sum)

    def collectResult(self, result):
        print("Result is", result)
        exit()


def main(args):
    # create Group of MyChare objects (there will be one object on each core)
    my_group = Group(MyChare)
    # invoke 'work' method of every element, passing number 3 as message
    my_group.work(3)


charm.start(main)

from charm4py import charm, Chare, Group, Reducer


def my_sum(contribs):
    return sum(contribs)

Reducer.addReducer(my_sum)


class Test(Chare):

    def __init__(self, future1, future2):
        self.contribute(1, Reducer.sum,    future1)
        self.contribute(1, Reducer.my_sum, future2)


def main(args):
    f1 = charm.createFuture()
    f2 = charm.createFuture()
    Group(Test, args=[f1, f2])
    assert f1.get() == charm.numPes()
    assert f2.get() == charm.numPes()
    exit()


charm.start(main)

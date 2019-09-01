from charm4py import Chare, Group, charm


class Hello(Chare):

    def SayHi(self):
        print('Hello World from element', self.thisIndex)


def main(args):
    # create Group of Hello objects (there will be one object on each core)
    hellos = Group(Hello)
    # call method 'SayHi' of all group members, wait for method to be invoked on all
    hellos.SayHi(awaitable=True).get()
    exit()


charm.start(main)

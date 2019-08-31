from charm4py import charm, Chare, Group

# This example creates a Group of Hello chares (a Group means that one instance
# will be created on each PE). A message is sent to the chare on the first PE,
# and then each chare forwards a modified message to the next chare.


class Hello(Chare):

    def sayHi(self, hello_num):
        print('Hi[' + str(hello_num) + '] from element', self.thisIndex)
        if self.thisIndex == charm.numPes() - 1:
            # we reached the last element
            print('All done')
            exit()
        else:
            # pass the hello message to the next element
            self.thisProxy[self.thisIndex + 1].sayHi(hello_num + 1)


def main(args):
    print('\nRunning Hello on', charm.numPes(), 'processors')
    # create a Group of Hello chares (there will be one chare per PE)
    group_proxy = Group(Hello)
    # send hello message to the first element
    group_proxy[0].sayHi(17)


charm.start(main)

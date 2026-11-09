from charm4py import charm, Chare, Group
import hello
import goodbye


class Main(Chare):

    def __init__(self, args):
        # create Group of chares of type hello.Hello
        hello_chares = Group(hello.Hello)
        # create Group of chares of type goodbye.Goodbye
        bye_chares = Group(goodbye.Goodbye)
        # add bye_chares proxy to globals of module hello on every process
        future1 = charm.thisProxy.updateGlobals({'bye_chares': bye_chares},
                                                module_name='hello', awaitable=True)
        # add mainchare proxy to globals of module goodbye on every process
        future2 = charm.thisProxy.updateGlobals({'mainProxy': self.thisProxy},
                                                module_name='goodbye', awaitable=True)
        charm.wait((future1, future2))
        # broadcast a message to the hello chares
        hello_chares.SayHi()

    def done(self):
        exit()


# Start a main chare of type Main. We specify to the charm runtime which
# modules contain Chare definitions. Note that the __main__ module is always
# searched for chare definitions, so we don't have to specify it
charm.start(Main)

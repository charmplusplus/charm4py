from charm4py import register, charm, Chare

# this will be set by Main chare
mainProxy = None


@register
class Goodbye(Chare):

    def SayGoodbye(self):
        if charm.myPe() < 10:
            print('Goodbye from PE', charm.myPe())
        # goodbye chares do an empty reduction. after the reduction completes,
        # the 'done' method of the mainchare will be called.
        # mainProxy is a global of this module, set previously from the mainchare
        self.reduce(mainProxy.done)

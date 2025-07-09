from charm4py import register, charm, Chare
import time

# this will be set by Main chare
bye_chares = None


@register
class Hello(Chare):

    def SayHi(self):
        if charm.myPe() < 10:
            print('Hello from PE', charm.myPe(), 'on', time.strftime('%c'))
        # call SayGoodbye method of the goodbye chare on my PE, bye_chares is
        # a global variable of this module, set previously from the mainchare
        bye_chares[charm.myPe()].SayGoodbye()

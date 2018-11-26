from charm4py import charm, Chare, Group
from charm4py import readonlies as ro
import hello, goodbye

class Main(Chare):
    def __init__(self, args):
        ro.mainProxy = self.thisProxy
        hellos  = Group(hello.Hello)
        ro.byes = Group(goodbye.Goodbye)
        hellos.SayHi()

    def done(self):
        exit()


charm.start(Main, modules=['hello', 'goodbye'])

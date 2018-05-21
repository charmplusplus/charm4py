from charmpy import charm, Chare, Group
from charmpy import readonlies as ro
import hello, goodbye

class Main(Chare):
    def __init__(self, args):
        ro.mainProxy = self.thisProxy
        hellos  = Group(hello.Hello)
        ro.byes = Group(goodbye.Goodbye)
        hellos.SayHi()

    def done(self):
        charm.exit()


charm.start(Main, modules=['hello', 'goodbye'])

from charmpy import charm, Mainchare, Group
from charmpy import readonlies as ro
import hello, goodbye

class Main(Mainchare):
    def __init__(self, args):
        ro.mainProxy = self.thisProxy
        hellos  = Group(hello.Hello)
        ro.byes = Group(goodbye.Goodbye)
        hellos.SayHi()

    def done(self):
        charm.exit()


charm.start(modules=['hello', 'goodbye'])

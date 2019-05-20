from charm4py import charm, Chare, Group
import hello
import goodbye


class Main(Chare):

    def __init__(self, args):
        hellos = Group(hello.Hello)
        byes = Group(goodbye.Goodbye)
        charm.thisProxy.updateGlobals({'mainProxy': self.thisProxy, 'byes': byes}, 'hello', ret=True).get()
        charm.thisProxy.updateGlobals({'mainProxy': self.thisProxy}, 'goodbye', ret=True).get()
        hellos.SayHi()

    def done(self):
        exit()


charm.start(Main, modules=['hello', 'goodbye'])

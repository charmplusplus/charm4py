from charmpy import charm, Mainchare
from charmpy import readonlies as ro

class Main(Mainchare):
    def __init__(self, args):
        ro.mainProxy = self.thisProxy
        hello = charm.HelloProxy.ckNew()
        ro.byeProxy = charm.GoodbyeProxy.ckNew()
        hello.SayHi()

    def done(self):
        charm.exit()


charm.start(modules=['hello', 'goodbye'])

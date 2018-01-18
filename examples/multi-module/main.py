from charmpy import charm, Mainchare, CkExit
from charmpy import readonlies as ro

class Main(Mainchare):
    def __init__(self, args):
        super(Main, self).__init__()

        ro.mainProxy = self.thisProxy
        hello = charm.HelloProxy.ckNew()
        ro.byeProxy = charm.GoodbyeProxy.ckNew()
        hello.SayHi()

    def done(self):
        CkExit()


charm.start(modules=['hello', 'goodbye'])

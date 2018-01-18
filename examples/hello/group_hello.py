from charmpy import charm, Mainchare, Group, CkMyPe, CkNumPes, CkExit
from charmpy import readonlies as ro


class Main(Mainchare):
  def __init__(self, args):
    super(Main,self).__init__()
    print("Running Hello on " + str(CkNumPes()) + " processors")
    grpProxy = charm.HelloProxy.ckNew()
    grpProxy[0].SayHi(17)
    ro.mainProxy = self.thisProxy

  def done(self):
    print("All done")
    CkExit()

class Hello(Group):
  def __init__(self):
    super(Hello,self).__init__()
    print("Hello " + str(self.thisIndex) + " created")

  def SayHi(self, hiNo):
    print("Hi[" + str(hiNo) + "] from element " + str(self.thisIndex))
    if self.thisIndex + 1 < CkNumPes():
      # Pass the hello on:
      self.thisProxy[self.thisIndex+1].SayHi(hiNo+1)
    else:
      ro.mainProxy.done()

# ---- start charm ----
charm.start([Main,Hello])

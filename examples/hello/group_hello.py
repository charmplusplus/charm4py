from charmpy import charm, Chare, Mainchare, Group
from charmpy import readonlies as ro


class Main(Mainchare):
  def __init__(self, args):
    print("Running Hello on " + str(charm.numPes()) + " processors")
    grpProxy = Group(Hello)
    grpProxy[0].SayHi(17)
    ro.mainProxy = self.thisProxy

  def done(self):
    print("All done")
    charm.exit()

class Hello(Chare):
  def __init__(self):
    print("Hello " + str(self.thisIndex) + " created")

  def SayHi(self, hiNo):
    print("Hi[" + str(hiNo) + "] from element " + str(self.thisIndex))
    if self.thisIndex + 1 < charm.numPes():
      # Pass the hello on:
      self.thisProxy[self.thisIndex+1].SayHi(hiNo+1)
    else:
      ro.mainProxy.done()

# ---- start charm ----
charm.start()

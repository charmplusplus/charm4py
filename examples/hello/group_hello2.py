from charmpy import charm, Mainchare, Group, CkMyPe, CkNumPes, CkExit
from charmpy import readonlies as ro


class HelloList:
  def __init__(self, hiNo):
    self.hiNo = hiNo
    self.hellos = []
  def addHello(self, msg): self.hellos.append(msg)
  def __str__(self):
    return "RESULT:\n" + "\n".join(self.hellos)

class Main(Mainchare):
  def __init__(self, args):
    print("Running Hello on " + str(CkNumPes()) + " processors")
    grpProxy = charm.HelloProxy.ckNew()
    grpProxy[0].SayHi(HelloList(17))
    ro.mainProxy = self.thisProxy

  def done(self, hellos):
    print("All done " + str(hellos))
    CkExit()

class Hello(Group):
  def __init__(self):
    print("Hello " + str(self.thisIndex) + " created")

  def SayHi(self, hellos):
    hellos.addHello("Hi[" + str(hellos.hiNo) + "] from element " + str(self.thisIndex))
    hellos.hiNo += 1
    if self.thisIndex + 1 < CkNumPes():
      # Pass the hello list on:
      self.thisProxy[self.thisIndex+1].SayHi(hellos)
    else:
      ro.mainProxy.done(hellos)


# ---- start charm ----
charm.start()

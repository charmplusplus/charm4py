from charmpy import charm, Mainchare, Group, CkExit, CkMyPe, CkNumPes, when, CkAbort
from charmpy import readonlies as ro

GRP_TO_SEND = 20

class Main(Mainchare):
  def __init__(self, args):
    super(Main,self).__init__()
    if CkNumPes() < 3: CkAbort("Run program with at least 3 PEs")
    grpProxy = charm.TestGProxy.ckNew()
    grpProxy.run()

class TestG(Group):
  def __init__(self):
    super(TestG,self).__init__()
    self.msgsRcvd = 0   # for PE 0
    self.current  = 1   # for PE 0
    self.msgsSent = 0   # for PEs != 0
    #print("Group constructed " + str(self.thisIndex))

  @when("current")
  def testWhen(self, id, msg):
    assert (CkMyPe() == 0) and (self.current == id) and (msg == "hi")
    print(str(id) + " " + str(self.msgsRcvd))
    self.msgsRcvd += 1
    if self.msgsRcvd >= GRP_TO_SEND:
      self.msgsRcvd = 0
      self.current += 1
      if self.current == CkNumPes(): CkExit()

  def run(self):
    if CkMyPe() == 0: return
    #print("Group " + str(self.thisIndex) + " sending msg " + str(self.msgsSent))
    self.thisProxy[0].testWhen(CkMyPe(), "hi")
    self.msgsSent += 1
    if self.msgsSent < GRP_TO_SEND: self.thisProxy[self.thisIndex].run()

charm.start()

from charm4py import charm, Chare, Group, when
from charm4py import readonlies as ro

GRP_TO_SEND = 20


class Test(Chare):

    def __init__(self):
        self.msgsRcvd = 0   # for PE 0
        self.current  = 1   # for PE 0
        self.msgsSent = 0   # for PEs != 0
        #print("Group constructed " + str(self.thisIndex))

    @when("self.current == id")
    def testWhen(self, id, msg):
        assert (charm.myPe() == 0) and (self.current == id) and (msg == "hi")
        print(str(id) + " " + str(self.msgsRcvd))
        self.msgsRcvd += 1
        if self.msgsRcvd >= GRP_TO_SEND:
            self.msgsRcvd = 0
            self.current += 1
            if self.current > ro.numParticipants:
                exit()

    def run(self):
        if charm.myPe() == 0 or charm.myPe() > ro.numParticipants: return
        #print("Group " + str(self.thisIndex) + " sending msg " + str(self.msgsSent))
        self.thisProxy[0].testWhen(charm.myPe(), "hi")
        self.msgsSent += 1
        if self.msgsSent < GRP_TO_SEND:
            self.thisProxy[self.thisIndex].run()


def main(args):
    if charm.numPes() < 3:
        charm.abort("Run program with at least 3 PEs")
    ro.numParticipants = min(charm.numPes()-1, 31)
    Group(Test).run()


charm.start(main)

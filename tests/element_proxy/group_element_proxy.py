"""
Tests the index proxy for a particular group element.
"""

from charmpy import charm, Mainchare, Group, CkMyPe, CkExit, CkNumPes, CkAbort

class Main(Mainchare):
    """
    Main chare.
    """

    def __init__(self, args):
        super(Main, self).__init__()
        if CkNumPes() < 3:
            CkAbort("Run program with at least 3 PEs")
        grp_proxy = charm.TestGroupProxy.ckNew()
        grp_proxy[0].start()

class TestGroup(Group):
    """
    A chare group to test the element proxy.
    """

    def __init__(self):
        super(TestGroup, self).__init__()
        self.count = 0

    def say(self, msg):
        """
        Helper method which is called by invoking the element proxy.
        This method is expected to be called on only the PE for
        which the proxy is created.
        """

        self.count += 1
        print("Say", msg, "on PE", CkMyPe())
        if self.count == 2:
            assert CkMyPe() == 2
            CkExit()

    def start(self):
        """
        Method which contains the started code.
        """

        proxy = self.thisProxy[2]
        self.thisProxy[1].say("hello")
        proxy.say("bye")
        proxy.say("bye")

# ---- start charm ----
charm.start()

"""
Tests the index proxy for a particular group element.
"""

from charmpy import charm, Chare, Mainchare, Group, CkMyPe, CkNumPes

class Main(Mainchare):
    """
    Main chare.
    """

    def __init__(self, args):
        if CkNumPes() < 3:
            charm.abort("Run program with at least 3 PEs")
        grp_proxy = Group(Test)
        grp_proxy[0].start()

class Test(Chare):
    """
    A chare group to test the element proxy.
    """

    def __init__(self):
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
            charm.exit()

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

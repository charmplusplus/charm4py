"""
Tests the index proxy for a particular array element.
"""

from charmpy import charm, Mainchare, Array, CkMyPe, CkExit

class Main(Mainchare):
    """
    Main chare.
    """

    def __init__(self, args):
        super(Main, self).__init__()
        arr_proxy = charm.TestArrayProxy.ckNew(6)
        arr_proxy[0].start()

class TestArray(Array):
    """
    A chare array to test the element proxy.
    """

    def __init__(self):
        super(TestArray, self).__init__()
        self.count = 0

    def say(self, msg):
        """
        Helper method which is called by invoking the element proxy.
        This method is expected to be called on only the chare for
        which the proxy is created.
        """

        self.count += 1
        print("Say", msg, "called on", self.thisIndex, "on PE", CkMyPe())
        if self.count == 2:
            assert self.thisIndex == (3,)
            CkExit()

    def start(self):
        """
        Method which contains the started code.
        """

        proxy = self.thisProxy[3]
        proxy.say("bye")
        proxy.say("bye")

# ---- start charm ----
charm.start()

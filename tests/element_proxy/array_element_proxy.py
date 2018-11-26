"""
Tests the index proxy for a particular array element.
"""

from charm4py import charm, Chare, Array


class Test(Chare):
    """
    A chare array to test the element proxy.
    """

    def __init__(self):
        self.count = 0

    def say(self, msg):
        """
        Helper method which is called by invoking the element proxy.
        This method is expected to be called on only the chare for
        which the proxy is created.
        """

        self.count += 1
        print("Say", msg, "called on", self.thisIndex, "on PE", charm.myPe())
        if self.count == 2:
            assert self.thisIndex == (3,)
            exit()

    def start(self):
        """
        Method which contains the started code.
        """

        proxy = self.thisProxy[3]
        proxy.say("bye")
        proxy.say("bye")


def main(args):
    arr_proxy = Array(Test, 6)
    arr_proxy[0].start()


charm.start(main)

from charm4py import charm, Chare, Array, expedited, Channel, EntryMethodOptions, coro


class Test(Chare):
    """
    A chare array to test the element proxy.
    """

    @coro
    def __init__(self):
        self.count = 0
        opts = EntryMethodOptions()
        opts.set_option(0x4)
        self.partner = Channel(self, self.thisProxy[(self.thisIndex[0] + 1) % 6], options=opts)
        self.partner.send(1)
        self.partner.recv()

    @expedited
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
        proxy = self.thisProxy[3]
        proxy.say("bye")
        proxy.say("bye")


def main(args):
    arr_proxy = Array(Test, 6)
    arr_proxy[0].start()


charm.start(main)

from charm4py import charm, Array, Chare, Reducer

CHARES_PER_PE = 4

def main(args):
    numChares = min(charm.numPes() * CHARES_PER_PE, 64)
    testProxy = Array(Test, numChares)

    f = charm.createFuture(senders=numChares)
    testProxy.getData(f)

    data = f.get()
    print("[Main] Received data: " + str(data))
    assert sorted(data) == list(range(numChares)), "Multi-futures failed!"
    print("[Main] All done.")
    exit()


class Test(Chare):
    def __init__(self):
        pass

    def getData(self, future):
        future.send(self.thisIndex[0])

charm.start(entry=main)

from charm4py import charm, Chare, Group, Array, Reducer, coro, Future
import sys


class Controller(Chare):

    @coro
    def start(self, workers, callback):
        f = Future()
        workers.work(f)
        result = f.get()
        callback(result)


class Worker(Chare):

    def work(self, callback):
        self.contribute(self.thisIndex, Reducer.sum, callback)


class CallbackReceiver(Chare):

    def __init__(self, main):
        self.main = main

    def getResult(self, result):
        #print('[' + str(charm.myPe()) + '] got result:', result)
        assert result == (charm.numPes() * (charm.numPes() - 1)) // 2
        self.main.workDone(self.thisIndex[0])

    def getResultBroadcast(self, result):
        #print('[' + str(charm.myPe()) + '] got result:', result)
        assert result == (charm.numPes() * (charm.numPes() - 1)) // 2
        self.contribute(1, Reducer.sum, self.main.workDone)


class Main(Chare):

    def __init__(self, args):
        if sys.version_info < (3, 0, 0):  # not supported in Python 2.7
            exit()
        assert charm.numPes() >= 4
        self.done = -1
        workers = Group(Worker)
        controllers = Array(Controller, charm.numPes())
        receivers = Array(CallbackReceiver, charm.numPes(), args=[self.thisProxy])
        workers.work(receivers[1].getResult)
        self.wait('self.done == 1')
        self.done = -1

        controllers[1].start(workers, receivers[2].getResult)
        self.wait('self.done == 2')
        self.done = -1

        controllers[2].start(workers, receivers.getResultBroadcast)
        self.wait('self.done == ' + str(charm.numPes()))
        self.done = -1

        f = Future()
        controllers[3].start(workers, f)
        assert f.get() == (charm.numPes() * (charm.numPes() - 1)) // 2

        exit()

    def workDone(self, src_info):
        self.done = src_info


charm.start(Main)

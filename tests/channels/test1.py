from charm4py import charm, Chare, Channel, coro, Future


class Test(Chare):

    def __init__(self, id):
        self.id = id

    @coro
    def work(self, mainProxy, other, done_fut):
        me = self.thisProxy[self.thisIndex]
        ch = Channel(self, remote=mainProxy)
        ch.send('hello from ' + str(self.id))

        ch = Channel(self, remote=me)
        ch.send('self ping', me)
        assert ch.recv() == ('self ping', me)

        ch = Channel(self, remote=other)
        ch.send(('hi from ' + str(self.id), me))
        data = ch.recv()
        assert data[0] == 'hi from ' + str((self.id + 1) % 2)
        assert data[1] == other
        done_fut()


class Main(Chare):

    def __init__(self, args):
        assert charm.numPes() >= 3
        done_fut = Future(2)
        chare0 = Chare(Test, onPE=1, args=[0])
        chare1 = Chare(Test, onPE=2, args=[1])
        chare0.work(self.thisProxy, chare1, done_fut)
        chare1.work(self.thisProxy, chare0, done_fut)
        ch0 = Channel(self, remote=chare0)
        ch1 = Channel(self, remote=chare1)
        assert ch0.recv() == 'hello from 0'
        assert ch1.recv() == 'hello from 1'
        done_fut.get()
        exit()


charm.start(Main)

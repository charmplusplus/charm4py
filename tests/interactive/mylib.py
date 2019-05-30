from charm4py import Chare, Reducer


class Worker(Chare):

    def work(self, cb, x=1):
        self.contribute(x, Reducer.sum, cb)

from charm4py import charm, Chare, Array, when
from charm4py import readonlies as ro
import random
import time

from charm4py import Options
Options.PROFILING = True

X, Y, Z = 3, 7, 11
WORKER_ITERS = 10
TASKS_PER_WORKER = 40
NUM_ITERS = 30
MAX_CHARES = 128

class Worker(Chare):

    def __init__(self, controller):
        self.ready      = True
        self.controller = controller

    @when("self.ready")
    def startWork(self, x, y, z):
        assert(self.ready)
        self.ready = False
        self.thisProxy[self.thisIndex].doWork(x, y, z)

    def doWork(self, x, y, z):
        assert(not self.ready)
        result = 0
        for _ in range(WORKER_ITERS): result += (x*y*z)
        self.thisProxy[self.thisIndex].workDone(result)

    def workDone(self, result):
        assert(not self.ready)
        assert(result == X*Y*Z*WORKER_ITERS)
        self.ready = True
        self.controller.taskDone()


class Controller(Chare):

    def __init__(self):
        self.tasks_completed = 0
        self.iteration = 0
        self.thisProxy.sendWork()

    def sendWork(self):
        for i in range(ro.num_workers):
            for _ in range(TASKS_PER_WORKER):
                ro.workers[i].startWork(X, Y, Z)

    def taskDone(self):
        self.tasks_completed += 1
        if self.tasks_completed == ro.num_workers * TASKS_PER_WORKER:
            self.tasks_completed = 0
            self.iteration += 1
            if self.iteration == NUM_ITERS:
              charm.printStats()
              exit()
            else:
              self.sendWork()


def main(args):
    ro.num_workers = min(charm.numPes() * 8, MAX_CHARES)
    controller = Chare(Controller, onPE=0)
    ro.workers = Array(Worker, ro.num_workers, args=[controller])


charm.start(main)

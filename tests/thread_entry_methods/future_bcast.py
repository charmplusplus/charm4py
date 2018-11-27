from charm4py import charm, Chare, Group, Array
import random
import time


class Test(Chare):

  def work(self, sleepTimes):
    if isinstance(self.thisIndex, tuple):
      time.sleep(sleepTimes[self.thisIndex[0]])
    else:
      time.sleep(sleepTimes[self.thisIndex])


def main(args):
    a = Array(Test, charm.numPes())
    sleepTimes = [random.uniform(0.5, 1.5) for i in range(charm.numPes())]
    # for some reason, work() runs on PE 0 before sending the broadcast msgs out
    # to the other PEs, so I set wait time to 0 on PE 0
    sleepTimes[0] = 0.0
    t0 = time.time()
    a.work(sleepTimes, ret=True).get()  # wait for broadcast to complete
    wait_time = time.time() - t0
    assert(wait_time >= max(sleepTimes))
    print(wait_time, max(sleepTimes))

    g = Group(Test)
    sleepTimes = [random.uniform(0.5, 1.5) for i in range(charm.numPes())]
    sleepTimes[0] = 0.0
    t0 = time.time()
    g.work(sleepTimes, ret=True).get()  # wait for broadcast to complete
    wait_time = time.time() - t0
    assert(wait_time >= max(sleepTimes))
    print(wait_time, max(sleepTimes))

    exit()


charm.start(main)

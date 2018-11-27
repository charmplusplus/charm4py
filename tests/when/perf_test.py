from charm4py import charm, Chare, Group, when
import random
import time

MAX_VALS = 10000
PHASE_NUM = 10

class Worker(Chare):

    def start(self, done_future):
        self.cur_id    = 0
        self.phase_cnt = 0
        self.done_future = done_future

    @when("self.cur_id == id")
    def recv_id(self, id):
        #if self.thisIndex == 0:
        #    return self.contribute(None, None, self.done_future)
        assert(id == self.cur_id)
        self.phase_cnt += 1
        if self.phase_cnt == PHASE_NUM:
            self.phase_cnt = 0
            self.cur_id += 1
            if self.cur_id == MAX_VALS:
                self.contribute(None, None, self.done_future)


def main(args):

    g = Group(Worker)

    random.seed(45782)
    ids = []
    for i in range(MAX_VALS):
        #for _ in range(PHASE_NUM):
            #ids.append(i)
        ids.append(i)
    random.shuffle(ids)

    done = charm.createFuture()
    g.start(done, ret=True).get()
    t0 = time.time()
    for id in ids:
        #g.recv_id(id)
        for _ in range(PHASE_NUM):
            g.recv_id(id)
    done.get()
    print("Elapsed=", time.time() - t0)
    exit()


charm.start(main)

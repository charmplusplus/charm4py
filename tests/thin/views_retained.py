"""Received numpy arrays kept alive across many messages (review open question,
SPEC-4c.md 7).

Run: lcrun -n 2 python tests/thin/views_retained.py [keep|copy|drop] [N] +pe 2

A received numpy array is a view into the runtime message, which the binding keeps
with CkDynMsgRetain until the array is garbage (ckdyn_py.cpp make_ndarray, MsgRef).
PE 1 sends N messages (default 10000), each with a 64 KB float64 array, to PE 0, in
windows of 500 with an acknowledgement per window. PE 0 keeps every received array
('keep'), keeps a copy and drops the view ('copy'), or keeps nothing ('drop'). The
test checks that every array arrives intact and prints the wall time and the
receiver's resident set size (current and peak) before and after.
"""
from charm4py import charm, Chare, Group, Future
import numpy as np
import resource
import subprocess
import os
import time

SIZE = 64 * 1024 // 8   # float64 elements in 64 KB
WINDOW = 500


def rss_mb():
    # current resident set size of this process (ps reports KB)
    out = subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())], capture_output=True, text=True).stdout
    return int(out.strip()) / 1024.0


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)   # bytes on macOS


class G(Chare):

    def setup(self, mode, n, done):
        self.mode = mode
        self.n = n
        self.done = done
        self.kept = []
        self.count = 0
        self.rss0 = rss_mb()

    def start(self):
        # on PE 1: send the first window
        self.sent = 0
        self.rng = np.random.default_rng(1)
        self.send_window()

    def send_window(self):
        end = min(self.sent + WINDOW, self.n)
        for i in range(self.sent, end):
            x = self.rng.random(SIZE)   # random: macOS compresses constant pages
            x[0] = x[-1] = i
            self.thisProxy[0].recv(i, x)
        self.sent = end

    def ack(self):
        if self.sent < self.n:
            self.send_window()

    def recv(self, i, x):
        assert x.shape == (SIZE,) and x[0] == i and x[-1] == i, i
        if self.mode == 'keep':
            self.kept.append(x)
        elif self.mode == 'copy':
            self.kept.append(x.copy())
        self.count += 1
        if self.count % WINDOW == 0:
            self.thisProxy[1].ack()
        if self.count == self.n:
            for j, y in enumerate(self.kept):
                assert y[0] == j and y[-1] == j, j
            self.done.send((self.rss0, rss_mb(), peak_mb()))


def main(args):
    mode = args[1] if len(args) > 1 else 'keep'
    n = int(args[2]) if len(args) > 2 else 10000
    assert charm.numPes() == 2, 'run with 2 PEs (lcrun -n 2 ... +pe 2)'
    g = Group(G)
    done = Future()
    g.setup(mode, n, done, awaitable=True).get()
    t0 = time.time()
    g[1].start()
    rss0, rss1, peak = done.get()
    dt = time.time() - t0
    print('views_retained mode=%s n=%d: %.2f s, %.1f us per message, receiver RSS %.0f MB -> %.0f MB '
          '(peak %.0f MB), payload kept %.0f MB'
          % (mode, n, dt, dt / n * 1e6, rss0, rss1, peak, (n * SIZE * 8 / 2**20) if mode != 'drop' else 0))
    print('views_retained: PASS')
    exit()


charm.start(main)

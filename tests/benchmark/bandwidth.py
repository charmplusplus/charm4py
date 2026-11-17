"""GPU bandwidth between two PEs: a window of device arrays in flight, then an ack.

Ported from charm4py PR #208 (Charm4py 1.x: channels with gpu_src_ptrs and posted
addresses) to the Charm4py 2 API. Element 0 (PE 0) sends -window arrays of each
size to element 1 (PE 1) back to back; element 1 acknowledges the window once all
of it has arrived; bandwidth is bytes * window * iterations / time. Modes (-m):

  staged      device to pinned host memory (one pinned buffer per message of the
              window), numpy arrays in the messages, host to device on the receiver
              (PR #208 mode 0)
  device      the cupy array as an argument (a DEVICE slot, device to device); the
              receiver gets a charm4py.DeviceBuffer allocated by the runtime. A post
              (Chare.post_buffer) covers one message per method, so a window is not
              posted; pingpong_gpu.py -m device-post measures posted receives
              (PR #208 mode 1 posted the receive addresses on a channel)
  persistent  charm.DevicePersistent: -window send buffers on element 0 put into
              -window registered buffers of element 1, all outstanding at once; when
              their futures are set, one message tells element 1, which acknowledges

Run: python -m charmrun.start +p2 bandwidth.py -m device
     srun --mpi=pmix -n 2 --gpus=2 --gpu-bind=none env CUDA_VISIBLE_DEVICES=0,1 \\
         python bandwidth.py -m persistent -min 16777216 -max 16777216 +pe 2
     python -m charmrun.start +p2 bandwidth.py --dry-run    (no GPU needed)
"""
from charm4py import charm, Chare, Array, Future, coro
from argparse import ArgumentParser
import time

MODES = ('staged', 'device', 'persistent')
WARMUP_ITERS = 3


class Block(Chare):

    def setup(self, mode, nbytes, window):
        import cupy
        import cupyx
        self.cp = cupy
        self.mode = mode
        self.window = window
        self.me = self.thisIndex[0]
        self.partner = self.thisProxy[1 - self.me]
        nbuf = window if mode == 'persistent' or (mode == 'staged' and self.me == 0) else 1
        if self.me == 0:
            self.d = [cupy.full(nbytes, 1, dtype=cupy.int8) for _ in range(nbuf)]
        else:
            self.d = [cupy.zeros(nbytes, dtype=cupy.int8) for _ in range(nbuf)]
        if mode == 'staged' and self.me == 0:
            self.h = [cupyx.empty_pinned(nbytes, dtype=cupy.int8) for _ in range(window)]
        cupy.cuda.Device().synchronize()
        self.got = 0
        self.wait_fut = None
        self.last = None
        if mode == 'persistent':
            self.p = [charm.DevicePersistent(a) for a in self.d]
            return [p.handle for p in self.p]
        return None

    def set_peer(self, hs):
        self.peer_h = hs

    def mode_of(self):
        return self.p[0].mode(self.peer_h[0])

    def notify(self):
        self.got += 1
        if self.wait_fut is not None:
            f, self.wait_fut = self.wait_fut, None
            f.send(None)

    def wait_partner(self):
        if self.got == 0:
            self.wait_fut = Future()
            self.wait_fut.get()
        self.got -= 1

    def recv(self, x):
        """ element 1: one message of the window """
        if self.mode == 'staged':
            self.d[0].set(x)
        self.last = x
        self.got += 1
        if self.got == self.window:
            self.got = 0
            self.partner.notify()

    def window_done(self):
        """ element 1, persistent mode: the window's puts have completed """
        self.partner.notify()

    @coro
    def run(self, niter):
        """ element 0 """
        t0 = time.perf_counter()
        for i in range(niter + WARMUP_ITERS):
            if i == WARMUP_ITERS:
                t0 = time.perf_counter()
            if self.mode == 'staged':
                for w in range(self.window):
                    self.d[w].get(out=self.h[w])
                    self.partner.recv(self.h[w])
            elif self.mode == 'device':
                for w in range(self.window):
                    self.partner.recv(self.d[0])
            else:
                futs = [self.p[w].put(self.peer_h[w]) for w in range(self.window)]
                for f in futs:
                    f.get()
                self.partner.window_done()
            self.wait_partner()
        return time.perf_counter() - t0

    def check(self):
        cp = self.cp
        if self.mode == 'device':
            return bool(cp.all(cp.asarray(self.last) == 1))
        return all(bool(cp.all(a == 1)) for a in self.d)

    def finish(self):
        if self.mode == 'persistent':
            for p in self.p:
                p.close()
        self.d = self.h = self.last = None


def parse(argv):
    p = ArgumentParser(description='GPU bandwidth between two PEs (Charm4py 2)')
    p.add_argument('-m', '--mode', choices=MODES, default='device')
    p.add_argument('-min', type=int, default=4096, help='smallest message in bytes (default 4096)')
    p.add_argument('-max', type=int, default=16 << 20, help='largest message in bytes (default 16 MiB)')
    p.add_argument('-window', type=int, default=8, help='messages in flight (default 8)')
    p.add_argument('-low-iter', type=int, default=200, help='iterations up to -threshold bytes')
    p.add_argument('-high-iter', type=int, default=20, help='iterations above -threshold bytes')
    p.add_argument('-threshold', type=int, default=1 << 20)
    p.add_argument('-csv', action='store_true', help='print comma-separated values')
    p.add_argument('--dry-run', action='store_true', help='print the sizes and exit (no GPU needed)')
    args, _ = p.parse_known_args([a for a in argv[1:] if not a.startswith('+')])
    return args


def main(argv):
    args = parse(argv)
    sizes = []
    s = args.min
    while s <= args.max:
        sizes.append(s)
        s *= 2
    if charm.numPes() < 2:
        print('bandwidth: needs 2 PEs')
        charm.exit(1)
        return
    print('bandwidth: mode %s, sizes %d..%d bytes, window %d, %d PEs'
          % (args.mode, args.min, args.max, args.window, charm.numPes()))
    if args.dry_run:
        print('dry run: %d sizes, iterations %s' % (len(sizes), [args.low_iter if n <= args.threshold
                                                                  else args.high_iter for n in sizes]))
        charm.exit()
        return
    from charm4py import _ckdyn
    try:
        import cupy   # noqa: F401
    except ImportError:
        print('bandwidth: cupy is not installed')
        charm.exit(1)
        return
    if args.mode != 'staged' and not _ckdyn.has_device_support():
        print('bandwidth: -m %s needs a CUDA build of Charm++' % args.mode)
        charm.exit(1)
        return

    blocks = Array(Block, 2)
    charm.awaitCreation(blocks)
    if args.csv:
        print('mode,bytes,window,iterations,bandwidth_MBps')
    else:
        print('%-12s %12s %8s %10s %16s' % ('mode', 'bytes', 'window', 'iterations', 'bandwidth (MB/s)'))
    ok = True
    for nbytes in sizes:
        niter = args.low_iter if nbytes <= args.threshold else args.high_iter
        h = [f.get() for f in [blocks[i].setup(args.mode, nbytes, args.window, ret=True) for i in range(2)]]
        if args.mode == 'persistent':
            blocks[0].set_peer(h[1], ret=True).get()
            if nbytes == sizes[0]:
                print('# persistent transfer mode: %s' % blocks[0].mode_of(ret=True).get())
        elapsed = blocks[0].run(niter, ret=True).get()
        good = blocks[1].check(ret=True).get()
        ok = ok and good
        blocks.finish(ret=True).get()
        bw = nbytes * args.window * niter / elapsed / 1e6
        if args.csv:
            print('%s,%d,%d,%d,%.1f' % (args.mode, nbytes, args.window, niter, bw))
        else:
            print('%-12s %12d %8d %10d %16.1f%s' % (args.mode, nbytes, args.window, niter, bw,
                                                  '' if good else '  BAD DATA'))
    print('bandwidth: %s' % ('data checked' if ok else 'DATA MISMATCH'))
    charm.exit(0 if ok else 1)


charm.start(main)

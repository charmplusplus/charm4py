"""GPU ping-pong between two PEs: one-way latency and bandwidth of device arrays.

Ported from charm4py PR #208 (Charm4py 1.x: channels with gpu_src_ptrs and posted
addresses) to the Charm4py 2 API. Element 0 and element 1 of a two-element array
(on PE 0 and PE 1) bounce a cupy int8 array of each size; one-way time is half the
round trip. Modes (-m):

  staged       device to pinned host memory, a numpy array in the message, host to
               device on the receiver (PR #208 mode 0, "host staging")
  device       the cupy array as an argument (a DEVICE slot, device to device); the
               receiver gets a charm4py.DeviceBuffer allocated by the runtime
  device-post  the same, into an array the receiver posted with Chare.post_buffer
               (PR #208 mode 1, GPU direct into a posted receive address)
  persistent   charm.DevicePersistent on both sides: put into the partner's
               registered buffer, then a small message that tells it the data is there

Every mode wakes the coroutine of the receiving element through a future, so the
per-message overhead of the loop is the same in all of them.

Run: python -m charmrun.start +p2 pingpong_gpu.py -m device
     srun --mpi=pmix -n 2 --gpus=2 --gpu-bind=none env CUDA_VISIBLE_DEVICES=0,1 \\
         python pingpong_gpu.py -m persistent -min 16777216 -max 16777216 +pe 2
     python -m charmrun.start +p2 pingpong_gpu.py --dry-run    (no GPU needed)
"""
from charm4py import charm, Chare, Array, Future, coro
from argparse import ArgumentParser
import time

MODES = ('staged', 'device', 'device-post', 'persistent')
WARMUP_ITERS = 10


class Ping(Chare):

    def setup(self, mode, nbytes, niter):
        import cupy
        import cupyx
        self.cp = cupy
        self.mode = mode
        self.me = self.thisIndex[0]
        self.partner = self.thisProxy[1 - self.me]
        self.d_send = cupy.full(nbytes, self.me + 1, dtype=cupy.int8)
        self.d_recv = cupy.zeros(nbytes, dtype=cupy.int8)
        if mode == 'staged':
            self.h_send = cupyx.empty_pinned(nbytes, dtype=cupy.int8)
        cupy.cuda.Device().synchronize()
        self.got = 0
        self.wait_fut = None
        self.last = None
        self.expect = niter + WARMUP_ITERS   # messages this element will receive
        if mode == 'device-post':
            self.post_buffer(self.recv, 0, self.d_recv)
        if mode == 'persistent':
            self.p_send = charm.DevicePersistent(self.d_send)
            self.p_recv = charm.DevicePersistent(self.d_recv)
            return self.p_recv.handle
        return None

    def where(self):
        return 'PE %d device %d' % (charm.myPe(), self.cp.cuda.Device().id)

    def set_peer(self, h):
        self.peer_h = h

    def mode_of(self):
        return self.p_send.mode(self.peer_h)

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
        if self.mode == 'staged':
            self.d_recv.set(x)          # host to device
        elif self.mode == 'device-post':
            assert x is self.d_recv
            self.expect -= 1
            if self.expect > 0:   # no post is left behind for the next size's setup
                self.post_buffer(self.recv, 0, self.d_recv)
        self.last = x
        self.notify()

    def send(self):
        if self.mode == 'staged':
            self.d_send.get(out=self.h_send)   # device to host, synchronous
            self.partner.recv(self.h_send)
        elif self.mode == 'persistent':
            self.p_send.put(self.peer_h).get()
            self.partner.notify()
        else:
            self.partner.recv(self.d_send)

    @coro
    def run(self, niter):
        t0 = time.perf_counter()
        for i in range(niter + WARMUP_ITERS):
            if i == WARMUP_ITERS:
                t0 = time.perf_counter()
            if self.me == 0:
                self.send()
                self.wait_partner()
            else:
                self.wait_partner()
                self.send()
        return time.perf_counter() - t0

    def check(self):
        """ the last data received is the partner's pattern """
        cp = self.cp
        if self.mode == 'device':
            got = cp.asarray(self.last)
        else:
            got = self.d_recv
        return bool(cp.all(got == 2 - self.me))

    def finish(self):
        if self.mode == 'persistent':
            self.p_send.close()
            self.p_recv.close()
        self.d_send = self.d_recv = self.last = None


def parse(argv):
    p = ArgumentParser(description='GPU ping-pong between two PEs (Charm4py 2)')
    p.add_argument('-m', '--mode', choices=MODES, default='device')
    p.add_argument('-min', type=int, default=8, help='smallest message in bytes (default 8)')
    p.add_argument('-max', type=int, default=16 << 20, help='largest message in bytes (default 16 MiB)')
    p.add_argument('-low-iter', type=int, default=1000, help='iterations up to -threshold bytes')
    p.add_argument('-high-iter', type=int, default=100, help='iterations above -threshold bytes')
    p.add_argument('-threshold', type=int, default=65536)
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
        print('pingpong_gpu: needs 2 PEs')
        charm.exit(1)
        return
    print('pingpong_gpu: mode %s, sizes %d..%d bytes, %d PEs' % (args.mode, args.min, args.max, charm.numPes()))
    if args.dry_run:
        print('dry run: %d sizes, iterations %s' % (len(sizes), [args.low_iter if n <= args.threshold
                                                                  else args.high_iter for n in sizes]))
        charm.exit()
        return
    from charm4py import _ckdyn
    try:
        import cupy   # noqa: F401
    except ImportError:
        print('pingpong_gpu: cupy is not installed')
        charm.exit(1)
        return
    if args.mode != 'staged' and not _ckdyn.has_device_support():
        print('pingpong_gpu: -m %s needs a CUDA build of Charm++' % args.mode)
        charm.exit(1)
        return

    pings = Array(Ping, 2)
    charm.awaitCreation(pings)
    if args.csv:
        print('mode,bytes,iterations,one_way_us,bandwidth_MBps')
    else:
        print('%-12s %12s %10s %16s %16s' % ('mode', 'bytes', 'iterations', 'one-way (us)', 'bandwidth (MB/s)'))
    ok = True
    for nbytes in sizes:
        niter = args.low_iter if nbytes <= args.threshold else args.high_iter
        h = [pings[i].setup(args.mode, nbytes, niter, ret=True) for i in range(2)]
        h = [f.get() for f in h]
        if nbytes == sizes[0]:
            where = [pings[i].where(ret=True).get() for i in range(2)]
            print('# element 0: %s; element 1: %s' % tuple(where))
        if args.mode == 'persistent':
            pings[0].set_peer(h[1], ret=True).get()
            pings[1].set_peer(h[0], ret=True).get()
            if nbytes == sizes[0]:
                print('# persistent transfer mode: %s' % pings[0].mode_of(ret=True).get())
        futs = [pings[i].run(niter, ret=True) for i in range(2)]
        elapsed = futs[0].get()
        futs[1].get()
        good = all(pings[i].check(ret=True).get() for i in range(2))
        ok = ok and good
        pings.finish(ret=True).get()
        one_way = elapsed / niter / 2
        if args.csv:
            print('%s,%d,%d,%.3f,%.1f' % (args.mode, nbytes, niter, one_way * 1e6, nbytes / one_way / 1e6))
        else:
            print('%-12s %12d %10d %16.3f %16.1f%s' % (args.mode, nbytes, niter, one_way * 1e6,
                                                    nbytes / one_way / 1e6, '' if good else '  BAD DATA'))
    print('pingpong_gpu: %s' % ('data checked' if ok else 'DATA MISMATCH'))
    charm.exit(0 if ok else 1)


charm.start(main)

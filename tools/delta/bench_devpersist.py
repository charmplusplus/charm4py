"""One-way time of a device-to-device transfer with persistent device buffers
(SPEC-7-device.md section 6), next to bench_zc_delta.py's DEVICE slot path.

Run: srun --mpi=pmix ... python bench_devpersist.py OP BYTES +pe 2 [+gpushm]
     OP is get or put; BYTES the buffer size in bytes.

Element 0 (PE 0) and element 1 (PE 1) each wrap a cupy array of BYTES in a
charm.DevicePersistent. Element 1 issues OP with element 0's handle and waits
for its future: one transfer is the time from the call to the future's value
(the done callback included). 3 warm-up transfers, then 20 timed; the median
is printed with the transfer mode. The contents are checked after the last one.
"""
from charm4py import charm, Chare, Array, coro
import statistics
import time

WARM, ROUNDS = 3, 20


class P(Chare):

    def open(self, n, seed):
        import cupy
        self.cp = cupy
        # the device HAPI chose for this PE (cupy would otherwise default to device 0)
        dev = cupy.cuda.runtime.getDevice()
        cupy.cuda.Device(dev).use()
        self.a = cupy.arange(n, dtype=cupy.float64) + seed
        cupy.cuda.Device().synchronize()
        self.p = charm.DevicePersistent(self.a)
        return self.p.handle

    def where(self):
        import cupy
        d = self.a.device.id
        return 'PE %d device %d (%s)' % (charm.myPe(), d, cupy.cuda.runtime.deviceGetPCIBusId(d))

    @coro
    def run(self, op, h, n):
        cupy = self.cp
        mode = self.p.mode(h)
        fn = self.p.get if op == 'get' else self.p.put
        times = []
        for i in range(WARM + ROUNDS):
            t0 = time.perf_counter()
            fn(h).get()
            if i >= WARM:
                times.append(time.perf_counter() - t0)
        ok = True
        if op == 'get':
            ok = float(self.a[n - 1]) == n - 1 + 100.0
        return mode, times, ok

    def check(self, n):
        return float(self.a[n - 1]) == n - 1 + 200.0


def main(args):
    op = args[1] if len(args) > 1 else 'get'
    nbytes = int(args[2]) if len(args) > 2 else 16 << 20
    n = nbytes // 8
    arr = Array(P, 2)
    charm.awaitCreation(arr)
    h0 = arr[0].open(n, 100.0, ret=True).get()
    arr[1].open(n, 200.0, ret=True).get()
    print('BENCH devices: %s; %s' % (arr[0].where(ret=True).get(), arr[1].where(ret=True).get()))
    mode, times, ok = arr[1].run(op, h0, n, ret=True).get()
    if op == 'put':
        ok = arr[0].check(n, ret=True).get()
    print('BENCH persistent op=%s bytes=%d mode=%s rounds=%d median_one_way_ms=%.3f mean_one_way_ms=%.3f '
          'min_one_way_ms=%.3f check=%s'
          % (op, nbytes, mode, ROUNDS, statistics.median(times) * 1e3,
             sum(times) / len(times) * 1e3, min(times) * 1e3, 'ok' if ok else 'BAD'))
    charm.exit()


charm.start(main)

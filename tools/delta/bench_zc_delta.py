"""One-way time of a float64 array between two PEs: BUFFER, NCPY or DEVICE path
(SPEC-3.md section 6). A variant of logs/step-3/bench_zc.py that takes the size and
reports the median.

Run: srun --mpi=pmix -N 2 --ntasks-per-node=1 ... python bench_zc_delta.py MODE BYTES +pe 2
     MODE is buffer, ncpy or device; BYTES is the array size in bytes.

Main (PE 0) sends the array to element 1 (PE 1), which sends the array it received
to element 0 (PE 0), which sets main's future: one round trip is two crossings
between the processes. 3 warm-up round trips, then 20 timed; one-way = round trip / 2.
buffer: zerocopy_threshold above the size (the array is copied into each message);
ncpy: threshold equal to the size (NCPY slot); device: a cupy array (DEVICE slot).
The contents are checked once, in the last round trip.
"""
from charm4py import charm, Chare, Array, Future
import numpy as np
import statistics
import time

WARM, ROUNDS = 3, 20


class B(Chare):

    def ping(self, x, f, check):
        self.thisProxy[0].pong(x, f, check)

    def pong(self, x, f, check):
        if check:
            if hasattr(x, '__cuda_array_interface__'):
                import cupy
                a = cupy.asarray(x)
                ok = float(a[a.shape[0] - 1]) == a.shape[0] - 1
            else:
                ok = x[x.shape[0] - 1] == x.shape[0] - 1
            f.send(bool(ok))
        else:
            f.send(True)


def main(args):
    mode = args[1] if len(args) > 1 else 'ncpy'
    nbytes = int(args[2]) if len(args) > 2 else 16 << 20
    n = nbytes // 8
    charm.options.zerocopy_threshold = 2 * nbytes + 1 if mode == 'buffer' else nbytes
    arr = Array(B, 2)
    charm.awaitCreation(arr)
    if mode == 'device':
        import cupy
        x = cupy.arange(n, dtype=cupy.float64)
        cupy.cuda.Device().synchronize()
    else:
        x = np.arange(n, dtype=np.float64)
    times = []
    ok = True
    for i in range(WARM + ROUNDS):
        last = i == WARM + ROUNDS - 1
        f = Future()
        t0 = time.perf_counter()
        arr[1].ping(x, f, last)
        r = f.get()
        if i >= WARM:
            times.append(time.perf_counter() - t0)
        if last:
            ok = r
    print('BENCH mode=%s bytes=%d rounds=%d median_one_way_ms=%.3f mean_one_way_ms=%.3f '
          'min_one_way_ms=%.3f check=%s'
          % (mode, nbytes, ROUNDS, statistics.median(times) / 2 * 1e3,
             sum(times) / len(times) / 2 * 1e3, min(times) / 2 * 1e3, 'ok' if ok else 'BAD'))
    charm.exit()


charm.start(main)

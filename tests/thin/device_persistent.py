"""Persistent device buffers: charm.DevicePersistent (SPEC-7-device.md sections 2 and 5).
Needs a CUDA or HIP build of Charm++ (_ckdyn.has_device_support()) and cupy; prints
SKIP and exits otherwise.

Run: python -m charmrun.start tests/thin/device_persistent.py +p2   (and +p1)
     add +gpushm for the IPC path between processes on one host

Element 0 and element 1 (another PE at +p2) each wrap a 16 MB cupy array:
- element 1 gets element 0's buffer through its handle (sent as an argument),
  checks it, writes a second pattern and puts it back; element 0 checks;
- a handle sent inside a list (pickled) works the same way;
- a second transfer while one is outstanding raises Charm4PyError;
- element 0 closes its buffer and opens it again; the new handle works;
- a get between two buffers on one element (copy in one process);
- the mode of each transfer is printed (copy, ipc or rdma).
"""
from charm4py import charm, Chare, Array, coro
from charm4py import _ckdyn

try:
    import cupy
except ImportError:
    cupy = None

N = 16 * 1024 * 1024 // 8   # 16 MB of float64


def pattern(seed):
    return cupy.arange(N, dtype=cupy.float64) * 0.5 + seed


class D(Chare):

    def open(self, seed):
        self.a = pattern(seed) if seed else cupy.zeros(N, dtype=cupy.float64)
        cupy.cuda.Device().synchronize()
        self.p = charm.DevicePersistent(self.a)
        return self.p.handle

    def check(self, seed):
        return bool(cupy.all(self.a == pattern(seed)))

    def reopen(self):
        self.p.close()
        assert self.p.closed
        self.p = charm.DevicePersistent(self.a)
        return self.p.handle

    def close(self):
        self.p.close()

    @coro
    def get_put(self, h, seed, seed2):
        mode = self.p.mode(h)
        assert self.p.get(h).get() is None
        assert bool(cupy.all(self.a == pattern(seed))), 'get: data differs'
        self.a[:] = pattern(seed2)
        cupy.cuda.Device().synchronize()
        assert self.p.put(h).get() is None
        return mode

    @coro
    def get_from_list(self, hs, seed):
        assert isinstance(hs, list) and isinstance(hs[0], charm.DevicePersistentHandle)
        self.p.get(hs[0]).get()
        return bool(cupy.all(self.a == pattern(seed)))

    @coro
    def outstanding(self, h):
        f = self.p.get(h)
        err = None
        try:
            self.p.get(h)
        except Exception as e:
            err = type(e).__name__
        f.get()
        try:
            self.p.close()
            self.p = charm.DevicePersistent(self.a)   # close is allowed again once the transfer is done
        except Exception as e:
            err = 'close after the transfer: ' + type(e).__name__
        return err

    @coro
    def local_pair(self, seed):
        b = pattern(seed)
        cupy.cuda.Device().synchronize()
        q = charm.DevicePersistent(b)
        mode = self.p.mode(q.handle)
        self.p.get(q.handle).get()
        q.close()
        return mode, bool(cupy.all(self.a == pattern(seed)))


def main(args):
    if cupy is None or not _ckdyn.has_device_support():
        print('SKIP device_persistent.py: %s' % ('cupy is not installed' if cupy is None
                                                 else 'this Charm++ has no CUDA or HIP support'))
        charm.exit()
        return
    arr = Array(D, 2)
    charm.awaitCreation(arr)
    h0 = arr[0].open(11.0, ret=True).get()
    arr[1].open(0, ret=True).get()
    assert isinstance(h0, charm.DevicePersistentHandle)
    mode = arr[1].get_put(h0, 11.0, 12.0, ret=True).get()
    assert arr[0].check(12.0, ret=True).get(), 'put: data differs on element 0'
    print('device_persistent: get and put between element 0 and element 1 passed, mode %s' % mode)

    assert arr[1].get_from_list([h0], 12.0, ret=True).get(), 'handle in a list: data differs'
    err = arr[1].outstanding(h0, ret=True).get()
    assert err == 'Charm4PyError', 'second transfer while one is outstanding: %s' % err
    print('device_persistent: handle in a list, outstanding-transfer error passed')

    h0b = arr[0].reopen(ret=True).get()
    mode2 = arr[1].get_put(h0b, 12.0, 13.0, ret=True).get()
    assert arr[0].check(13.0, ret=True).get(), 'after reopen: data differs'
    lm, ok = arr[0].local_pair(14.0, ret=True).get()
    assert ok and lm == 'copy', 'local pair: mode %s ok %s' % (lm, ok)
    arr.close(ret=True).get()
    print('device_persistent: close, reopen, get and put passed (mode %s); local pair (copy) passed' % mode2)
    charm.exit()


charm.start(main)

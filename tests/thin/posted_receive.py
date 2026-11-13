"""Posted receives: Chare.post_buffer (SPEC-7-device.md sections 3 and 5).

Run: python -m charmrun.start tests/thin/posted_receive.py +p1   (and +p2)
     python -m charmrun.start tests/thin/posted_receive.py +p1 -mismatch   (must abort)

The zero-copy threshold is lowered to 64 KB. Element 1 (on another PE at +p2)
posts a numpy array for an argument, and the method receives that same object:
- a 2 MB array (an NCPY slot): `x is posted`, contents, the post is consumed;
- the same with ret=True, whose message carries a block future before the
  arguments (the post counts arguments, not slots);
- the second argument of a method;
- an array below the threshold (a BUFFER slot): copied into the posted array
  by the binding, identity kept, post consumed;
- a broadcast to both elements, each with its own post (copied);
- errors: a second post for the same argument, a slot out of range, a method
  of another object, a stale object after migration (Charm4PyError);
- post then migrate (+p2): the post is dropped and the next message allocates;
- -mismatch: a 1 MB post for a 2 MB array aborts with both lengths.
With cupy and a CUDA build of Charm++ the device phases run too: a cupy array
posted and received directly (identity, contents, same device pointer), and a
broadcast of a cupy array (pickled) into posted cupy arrays.
"""
import sys
from charm4py import charm, Chare, Array, Future
from charm4py import _ckdyn
import numpy as np

try:
    import cupy
except ImportError:
    cupy = None

N = 2 * 1024 * 1024 // 8   # 2 MB of float64
STALE = {}


def pattern(seed, n=N):
    return np.arange(n, dtype=np.float64) * 0.5 + seed


class R(Chare):

    def __init__(self):
        self.posted = None
        self.mig_fut = None

    def post(self, method, slot, n, device=False):
        if device:
            self.posted = cupy.zeros(n, dtype=cupy.float64)
        else:
            self.posted = np.zeros(n, dtype=np.float64)
        self.post_buffer(getattr(self, method), slot, self.posted)
        return _ckdyn.posted_count(self)

    def recv(self, x, seed):
        assert x is self.posted, 'recv: got a %s, not the posted array' % type(x).__name__
        assert np.array_equal(x, pattern(seed, x.shape[0])), 'recv: data differs'
        return _ckdyn.posted_count(self)

    def status(self, seed):
        return _ckdyn.posted_count(self), bool(np.array_equal(self.posted, pattern(seed)))

    def recv2(self, a, x, seed):
        assert a == 'tag'
        assert x is self.posted, 'recv2: not the posted array'
        assert np.array_equal(x, pattern(seed)), 'recv2: data differs'
        return _ckdyn.posted_count(self)

    def bc(self, x, seed):
        return bool(x is self.posted and np.array_equal(x, pattern(seed, x.shape[0])))

    def drecv(self, x, seed):
        assert x is self.posted, 'drecv: got a %s, not the posted cupy array' % type(x).__name__
        assert bool(cupy.all(x == cupy.asarray(pattern(seed, x.shape[0])))), 'drecv: data differs'
        return _ckdyn.posted_count(self)

    def dbc(self, x, seed):
        return bool(x is self.posted and bool(cupy.all(x == cupy.asarray(pattern(seed, x.shape[0])))))

    def errors(self):
        out = []
        self.posted = np.zeros(N)
        self.post_buffer(self.recv, 0, self.posted)
        for what, f in (('second post', lambda: self.post_buffer(self.recv, 0, np.zeros(N))),
                        ('slot out of range', lambda: self.post_buffer(self.recv, 2, np.zeros(N))),
                        ('another object', lambda: self.post_buffer(R.recv.__get__(object.__new__(R)), 0, np.zeros(N)))):
            try:
                f()
                out.append(what + ': no error')
            except Exception as e:
                if type(e).__name__ != 'Charm4PyError':
                    out.append(what + ': ' + type(e).__name__ + ': ' + str(e))
        return out

    def post_and_migrate(self, to, fut):
        self.posted = np.zeros(N)
        self.post_buffer(self.recv_after, 0, self.posted)
        STALE['obj'] = self
        self.mig_fut = fut
        self.migrate(to)

    def migrated(self):
        if self.mig_fut is not None:
            f, self.mig_fut = self.mig_fut, None
            f.send(charm.myPe())

    def recv_after(self, x, seed):
        assert x is not self.posted, 'recv_after: the post survived migration'
        assert np.array_equal(x, pattern(seed)), 'recv_after: data differs'
        assert not self.posted.any(), 'recv_after: the dropped post was written'
        return _ckdyn.posted_count(self)


def stale_post():
    obj = STALE.pop('obj')
    try:
        obj.post_buffer(obj.recv_after, 0, np.zeros(N))
    except Exception as e:
        return type(e).__name__
    return 'no error'


def main(args):
    charm.options.zerocopy_threshold = 64 * 1024
    arr = Array(R, 2)
    charm.awaitCreation(arr)
    e1 = arr[1]

    if '-mismatch' in args:
        assert e1.post('recv', 0, N // 2, ret=True).get() == 1
        e1.recv(pattern(1.0), 1.0, ret=True).get()
        print('posted_receive -mismatch: no abort (FAIL)')
        charm.exit(1)
        return

    # NCPY slot into the posted array
    assert e1.post('recv', 0, N, ret=True).get() == 1
    assert e1.recv(pattern(1.0), 1.0, ret=True).get() == 0, 'post not consumed'
    # ret=True above put a block future before the arguments; now without one
    assert e1.post('recv', 0, N, ret=True).get() == 1
    e1.recv(pattern(2.0), 2.0)   # a plain send: no block future before the argument
    for i in range(500):   # the NCPY transfer can let a later message overtake it
        st = e1.status(2.0, ret=True).get()
        if st[0] == 0:
            break
        charm.sleep(0.01)
    assert st == (0, True), 'plain send into the post failed: %s' % (st,)
    # second argument
    assert e1.post('recv2', 1, N, ret=True).get() == 1
    assert e1.recv2('tag', pattern(3.0), 3.0, ret=True).get() == 0
    # below the threshold: BUFFER, copied by the binding
    assert e1.post('recv', 0, 1000, ret=True).get() == 1
    assert e1.recv(pattern(4.0, 1000), 4.0, ret=True).get() == 0, 'BUFFER post not consumed'
    # a broadcast (copied) into each element's post
    assert arr.post('bc', 0, N, ret=True).get() == [1, 1]
    assert arr.bc(pattern(5.0), 5.0, ret=True).get() == [True, True], 'broadcast into posts failed'
    # errors
    errs = e1.errors(ret=True).get()
    assert errs == [], 'errors: ' + '; '.join(errs)
    e1.recv(pattern(6.0), 6.0, ret=True).get()   # consumes the post made by errors()
    print('posted_receive: host posts passed (NCPY, ret=True, second argument, BUFFER, broadcast, errors)')

    if charm.numPes() > 1:
        mf = Future()
        arr[0].post_and_migrate(1, mf)
        assert mf.get() == 1
        assert stale_post() == 'Charm4PyError', 'post_buffer on a migrated-away object did not raise'
        assert arr[0].recv_after(pattern(7.0), 7.0, ret=True).get() == 0
        print('posted_receive: post then migrate passed (post dropped, stale object refused)')
    else:
        print('posted_receive: post then migrate skipped (one PE)')

    if cupy is not None and _ckdyn.has_device_support():
        assert e1.post('drecv', 0, N, True, ret=True).get() == 1
        x = cupy.asarray(pattern(8.0))
        cupy.cuda.Device().synchronize()
        assert e1.drecv(x, 8.0, ret=True).get() == 0, 'device post not consumed'
        assert arr.post('dbc', 0, N, True, ret=True).get() == [1, 1]
        assert arr.dbc(cupy.asarray(pattern(9.0)), 9.0, ret=True).get() == [True, True]
        print('posted_receive: device posts passed (DEVICE slot into the posted cupy array, broadcast)')
    else:
        print('SKIP posted_receive device phases: %s' % ('cupy is not installed' if cupy is None
                                                         else 'this Charm++ has no CUDA or HIP support'))
    charm.exit()


charm.start(main)

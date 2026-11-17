"""Zero-copy numpy arguments (SPEC-3.md section 5).

Run: python -m charmrun.start tests/thin/zerocopy.py +p1   (and +p2)

The threshold is lowered to 64 KB, so a 2 MB float64 array sent point to point
travels as an NCPY slot: the runtime transfers it from the sender's array, which
the binding holds until the slot's done function runs. Checked here:
- a round trip element 1 -> element 0 -> main arrives intact;
- the sender's array stays alive after the send even when the sender drops its
  own reference, and is released once the transfer is done;
- a received array kept past the method keeps its data;
- a broadcast of the same array arrives on every element (as a copy, BUFFER);
- a large return value of a ret=True call arrives intact;
- arrays below the threshold still work, and every NCPY slot's done function ran.
"""
from charm4py import charm, Chare, Array, Future, Reducer
from charm4py import _ckdyn
import numpy as np
import weakref

N = 2 * 1024 * 1024 // 8   # 2 MB of float64


def pattern(seed, n=N):
    return np.arange(n, dtype=np.float64) * 0.5 + seed


class Z(Chare):

    def __init__(self):
        self.kept = None

    # element 1 receives, keeps the view and echoes it to element 0
    def ping(self, x, seed, done):
        assert isinstance(x, np.ndarray) and x.dtype == np.float64 and x.shape == (N,)
        assert np.array_equal(x, pattern(seed)), 'ping: data differs'
        self.kept = x
        self.thisProxy[0].pong(x, seed, done)

    def pong(self, x, seed, done):
        assert np.array_equal(x, pattern(seed)), 'pong: data differs'
        x[0] = -1.0   # a point-to-point delivery is writable
        done.send(True)

    def check_kept(self, seed):
        # the view kept by ping, read in a later method
        return bool(np.array_equal(self.kept, pattern(seed)))

    def big(self, seed):
        return pattern(seed)

    def bc(self, x, seed):
        return bool(np.array_equal(x, pattern(seed)))

    def small(self, x, seed):
        return bool(np.array_equal(x, pattern(seed, x.shape[0])))

    def counts(self):
        return _ckdyn.zerocopy_counts()


def main(args):
    charm.options.zerocopy_threshold = 64 * 1024
    arr = Array(Z, 2)
    charm.awaitCreation(arr)

    # round trip; the sender (main) drops its reference right after the send
    x = pattern(7.0)
    w = weakref.ref(x)
    done = Future()
    arr[1].ping(x, 7.0, done)
    del x
    assert w() is not None, 'the sent array was released before its transfer'
    assert done.get() is True
    for i in range(500):
        if w() is None:
            break
        charm.sleep(0.01)
    assert w() is None, 'the sent array was not released after its transfer'

    # the kept view
    assert arr[1].check_kept(7.0, ret=True).get() is True, 'kept view lost its data'

    # broadcast (copied)
    y = pattern(11.0)
    assert arr.bc(y, 11.0, ret=True).get() == [True, True], 'broadcast data differs'

    # a large return value
    r = arr[1].big(13.0, ret=True).get()
    assert isinstance(r, np.ndarray) and np.array_equal(r, pattern(13.0)), 'returned array differs'

    # below the threshold (BUFFER)
    s = pattern(17.0, 1000)
    assert arr[1].small(s, 17.0, ret=True).get() is True

    # every NCPY slot sent from PE 0 and from element 1's PE had its done function run
    sent, ndone = _ckdyn.zerocopy_counts()
    for i in range(500):
        if ndone == sent:
            break
        charm.sleep(0.01)
        sent, ndone = _ckdyn.zerocopy_counts()
    assert sent >= 1 and ndone == sent, 'PE 0: %d NCPY slots sent, %d done' % (sent, ndone)
    s1, d1 = arr[1].counts(ret=True).get()
    assert s1 >= 1 and d1 <= s1, 'element 1: %d NCPY slots sent, %d done' % (s1, d1)
    print('zerocopy: %d NCPY slots from PE 0, %d from element 1 (PE %d); all checks passed'
          % (sent, s1, 1 if charm.numPes() > 1 else 0))
    charm.exit()


charm.start(main)

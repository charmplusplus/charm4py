"""Stresses reference counting: references created on many PEs, passed nested
in arguments and results, borrowed by actors and dropped while copies are in
flight. Every value must still arrive (nothing freed early) and, on charm4py,
the object store must end up empty (nothing leaked).

    python -m charmrun.start +p8 -m charm4py.ray.launch tests/ray_compat/refcount_stress.py
    python tests/ray_compat/refcount_stress.py          # real Ray (values only)
"""

import random
import sys
import time

import numpy as np
import ray


@ray.remote
def make(i, n):
    return np.full(n, i, dtype=np.int64)


@ray.remote
def wrap(refs):
    # returns references it was given (nested), plus a new one
    return refs + [make.remote(-1, 10)]


@ray.remote
def total(refs):
    return int(sum(int(a[0]) for a in ray.get(refs)))


@ray.remote
class Keeper:
    def __init__(self):
        self.kept = []

    def keep(self, refs):
        self.kept.extend(refs)
        if len(self.kept) > 6:
            self.kept = self.kept[-6:]
        return len(self.kept)

    def sum_kept(self):
        return int(sum(int(a[0]) for a in ray.get(self.kept)))

    def clear(self):
        self.kept = []


def main(rounds):
    ray.init(num_cpus=4)
    rng = random.Random(0)
    keepers = [Keeper.remote() for _ in range(3)]
    for r in range(rounds):
        refs = [make.remote(i, rng.choice([5, 200, 5000])) for i in range(8)]
        wrapped = wrap.remote(refs[:4])          # nested in an argument and a result
        inner = ray.get(wrapped)                  # references received from a task
        expect = sum(range(4)) - 1
        assert ray.get(total.remote(inner)) == expect, r
        keeper = keepers[r % 3]
        ray.get(keeper.keep.remote(refs[4:]))     # borrowed by an actor
        put = ray.put([refs[0], np.arange(3000)])  # nested in a put object
        del refs, inner, wrapped
        if rng.random() < 0.5:
            ready, not_ready = ray.wait([make.remote(1, 50000)], timeout=0)
            del ready, not_ready                  # dropped while possibly in flight
        assert int(ray.get(ray.get(put)[0])[0]) == 0, r
        del put
        assert ray.get(keeper.sum_kept.remote()) >= 0
    for keeper in keepers:
        ray.get(keeper.clear.remote())
    ok = True
    if getattr(ray, '__charm4py__', False):
        deadline = time.time() + 30
        while time.time() < deadline:
            left = sum(o for o, _ in ray._object_counts())
            if left == 0:
                break
            time.sleep(0.2)
        print('objects left:', left, ray._object_counts(), flush=True)
        ok = left == 0
    print(('PASS' if ok else 'FAIL') + ' refcount stress, ' + str(rounds) + ' rounds',
          flush=True)
    ray.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 200))

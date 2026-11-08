# Standalone test of the nanobind object store (src/objstore_py.cpp).
# Usage: PYTHONPATH=<build dir> python test_objstore.py
import sys, types
import numpy as np

# Fake charm4py.charm providing myPe/numPes (the store imports it lazily).
MY_PE, NUM_PES = 1, 4
fake_pkg = types.ModuleType('charm4py')
fake_pkg.charm = types.SimpleNamespace(myPe=lambda: MY_PE, numPes=lambda: NUM_PES)
sys.modules['charm4py'] = fake_pkg

from _objstore_test import MessageBuffer, CObjectStore

class FakeProxy:
    """proxy[pe].method(*args) appends (pe, method, args) to self.calls."""
    def __init__(self):
        self.calls = []
    def __getitem__(self, pe):
        assert type(pe) is int
        proxy = self
        class Elem:
            def __getattr__(self, name):
                return lambda *args: proxy.calls.append((pe, name, args))
        return Elem()
    def take(self):
        c, self.calls = self.calls, []
        return c

def ok(name):
    print('PASS', name)

# --- MessageBuffer
mb = MessageBuffer()
m1, m2 = object(), object()
mb.insert([10, 11], m1)
mb.insert(np.array([11], dtype=np.uint64), m2)
assert mb.check(99) == []
assert mb.check(10) == []
r = mb.check(11)
assert len(r) == 2 and r[0] is m1 and r[1] is m2
assert mb.check(11) == []
m3 = object()
mb.insert((5, 5), m3)  # duplicate id counts twice
assert mb.check(5) == [m3]
ok('MessageBuffer insert/check')

# --- lookup_object / insert_object_small / delete_object
p = FakeProxy(); s = CObjectStore(p)
assert s.lookup_object(7) is None
o = ['small']
s.insert_object_small(7, o)
assert s.lookup_object(7) is o
s.insert_object_small(7, ['other'])  # existing: ignored
assert s.lookup_object(7) is o
assert s.lookup_location(7) == MY_PE
s.delete_object(7); assert s.lookup_object(7) is None
s.delete_object(7)
assert p.take() == []
ok('lookup_object/insert_object_small/delete_object')

# --- lookup_location with fetch, and replica round robin
p = FakeProxy(); s = CObjectStore(p)
assert s.lookup_location(9) == -1
assert p.take() == [(9 % NUM_PES, 'request_location', (9, MY_PE))]
assert s.lookup_location(9, fetch=False) == -1 and p.take() == []
s.update_location(9, 2); s.update_location(9, 3)
assert [s.lookup_location(9) for _ in range(4)] == [2, 3, 2, 3]
ok('lookup_location fetch/round robin')

# --- delete_remote_objects
s.delete_remote_objects(9)
assert p.take() == [(2, 'delete_object', (9,)), (3, 'delete_object', (9,))]
s.delete_remote_objects(12345); assert p.take() == []
ok('delete_remote_objects')

# --- request_location buffering -> check_loc_requests_buffer via update_location
p = FakeProxy(); s = CObjectStore(p)
s.request_location(20, 3)   # unknown: fetch to home + buffer
s.request_location(20, 2)
assert p.take() == [(0, 'request_location', (20, MY_PE))] * 2
s.update_location(20, 1)
c = p.take()
assert len(c) == 1 and c[0][0] == 1 and c[0][1] == 'bulk_request_location'
arr = c[0][2][1]
assert type(arr) is np.ndarray and arr.dtype == np.int32 and list(arr) == [3, 2]
s.update_location(20, 0); assert p.take() == []   # not a new entry
s.request_location(20, 3)
c = p.take()   # location list is [1, 0]; the replica counter picks one
assert len(c) == 1 and c[0][:2] == (3, 'update_location') and c[0][2][0] == 20 and c[0][2][1] in (0, 1)
ok('request_location / check_loc_requests_buffer')

# --- request_location_object buffering -> check_obj_loc_requests_buffer
p = FakeProxy(); s = CObjectStore(p)
s.request_location_object(30, 2)
s.request_location_object(30, 3)
assert p.take() == []
s.insert_object_small(30, 'v')   # local object: location MY_PE
c = p.take()
assert len(c) == 1 and c[0][:2] == (MY_PE, 'bulk_send_object')
arr = c[0][2][1]
assert type(arr) is np.ndarray and arr.dtype == np.int32 and list(arr) == [2, 3]
s.request_location_object(30, 2)  # object present: send it directly
assert p.take() == [(2, 'receive_remote_object', (30, 'v'))]
s2 = CObjectStore(p); s2.update_location(31, 3)
s2.request_location_object(31, 0)
assert p.take() == [(3, 'request_object', (31, 0))]
assert {s2.lookup_location(31), s2.lookup_location(31)} == {3, 0}
ok('request_location_object / check_obj_loc_requests_buffer')

# --- request_object buffering -> receive_remote_object -> check_obj_requests_buffer
p = FakeProxy(); s = CObjectStore(p)
s.request_object(40, 0); s.request_object(40, 2)
assert p.take() == []
big = bytearray(4096)
s.receive_remote_object(40, big)
assert s.lookup_object(40) is big
assert p.take() == [(0, 'receive_remote_object', (40, big)),
                    (2, 'receive_remote_object', (40, big))]
s.receive_remote_object(40, big); assert p.take() == []  # buffer cleared
s.request_object(40, 3)
assert p.take() == [(3, 'receive_remote_object', (40, big))]
ok('request_object / check_obj_requests_buffer')

# --- bulk_send_object with a numpy int32 array
s.bulk_send_object(40, np.array([1, 2], dtype=np.int32))
assert p.take() == [(1, 'receive_remote_object', (40, big)),
                    (2, 'receive_remote_object', (40, big))]
ok('bulk_send_object')

# --- create_object small and large
p = FakeProxy(); s = CObjectStore(p)
s.create_object(50, 7)
assert p.take() == [(50 % NUM_PES, 'insert_object_small', (50, 7))]
assert s.lookup_object(50) is None
s.create_object(51, big)
assert s.lookup_object(51) is big
assert p.take() == [(51 % NUM_PES, 'update_location', (51, MY_PE))]
ok('create_object')

# --- reference counting: stored objects keep their referents alive
import gc, weakref
class Obj: pass
x = Obj(); w = weakref.ref(x)
s.receive_remote_object(60, x); del x; gc.collect()
assert w() is not None
s.delete_object(60); gc.collect()
assert w() is None
ok('refcounts')
# The fake charm4py lambdas hold this module's globals; drop the nanobind
# instances explicitly so nanobind's shutdown leak check stays quiet.
del mb, s, s2
# --- a cycle through the proxy is collectable
class CycProxy:
    pass
cp = CycProxy(); cp.store = CObjectStore(cp); wc = weakref.ref(cp)
del cp; gc.collect()
assert wc() is None
print('PASS proxy cycle collected')
print('ALL PASS')

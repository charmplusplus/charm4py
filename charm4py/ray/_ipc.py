"""Serialization between a Charm4py host and its worker processes.

References must not cross this boundary through ObjectRef's own pickling: that
runs Charm4py's borrower bookkeeping, which assumes every holder is a Charm PE.
Instead the host gives each reference it sends a token (its object id) and keeps
the real ObjectRef; the worker holds an inert stand-in, and when the token comes
back the host substitutes its own ObjectRef again. Actor handles coming from a
worker are mapped to one host-side handle per worker-side handle, so each keeps
its own call order."""

import io
import pickle

from .object_ref import ObjectRef

PROTOCOL = pickle.HIGHEST_PROTOCOL


class _HostPickler(pickle.Pickler):

    def __init__(self, buf, side):
        super().__init__(buf, PROTOCOL)
        self.side = side

    def persistent_id(self, o):
        if type(o) is ObjectRef:
            self.side.refs[o.store_id] = o
            return ('ref', o.store_id)
        return None


class _HostUnpickler(pickle.Unpickler):

    def __init__(self, buf, side):
        super().__init__(buf)
        self.side = side

    def persistent_load(self, pid):
        if pid[0] == 'ref':
            return self.side.refs[pid[1]]
        if pid[0] == 'actor':
            handle = self.side.handles.get(pid[1])
            if handle is None:
                from .actor import ActorHandle
                handle = self.side.handles[pid[1]] = ActorHandle._from_state(pid[2])
            return handle
        raise pickle.UnpicklingError('unknown persistent id ' + repr(pid))


class HostSide(object):
    """Per-worker translation tables, kept by the host."""

    def __init__(self):
        self.refs = {}       # object id -> host ObjectRef sent to the worker
        self.handles = {}    # worker handle id -> host ActorHandle

    def dumps(self, obj):
        buf = io.BytesIO()
        _HostPickler(buf, self).dump(obj)
        return buf.getvalue()

    def loads(self, data):
        return _HostUnpickler(io.BytesIO(data), self).load()


def stand_in_ref(store_id):
    """An inert ObjectRef for a worker process: it carries the id only, and it
    neither pickles nor deletes through Charm4py."""
    ref = ObjectRef.__new__(ObjectRef)
    ref.store_id = store_id
    ref.fid = store_id & 0xffffffff
    ref.src = store_id >> 32
    ref.store = True
    ref.parent = None
    ref.num_borrowers = 0
    ref.borrow_depth = 0
    ref._requested = False
    ref.gr = None
    ref.nvals = 1
    ref.values = []
    ref.blocked = False
    ref.gotvalues = False
    ref.error = None
    return ref


class _WorkerPickler(pickle.Pickler):

    def __init__(self, buf):
        super().__init__(buf, PROTOCOL)

    def persistent_id(self, o):
        if type(o) is ObjectRef:
            return ('ref', o.store_id)
        from .actor import ActorHandle
        if type(o) is ActorHandle:
            return ('actor', o._uid, o._state())
        return None


class _WorkerUnpickler(pickle.Unpickler):

    def __init__(self, buf, side):
        super().__init__(buf)
        self.side = side

    def persistent_load(self, pid):
        if pid[0] == 'ref':
            ref = self.side.refs.get(pid[1])
            if ref is None:
                ref = self.side.refs[pid[1]] = stand_in_ref(pid[1])
            return ref
        raise pickle.UnpicklingError('unknown persistent id ' + repr(pid))


class WorkerSide(object):
    """Translation in the worker process."""

    def __init__(self):
        self.refs = {}       # object id -> stand-in ObjectRef

    def dumps(self, obj):
        buf = io.BytesIO()
        _WorkerPickler(buf).dump(obj)
        return buf.getvalue()

    def loads(self, data):
        return _WorkerUnpickler(io.BytesIO(data), self).load()

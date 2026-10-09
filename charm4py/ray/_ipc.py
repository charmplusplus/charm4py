"""Serialization between a Charm4py host and its worker processes.

Values made by user code -- arguments, results, errors, objects put in the
store -- are serialized once, in the worker that makes them, into an Opaque.
Charm4py processes store and forward an Opaque without deserializing it, so they
never import the user's modules (vLLM, torch, ...); only the worker that uses
the value deserializes it. (Ray's raylet does not deserialize objects either.)

References must not cross the worker boundary through ObjectRef's own pickling:
that runs Charm4py's borrower bookkeeping, which assumes every holder is a Charm
PE. Instead the host gives each reference it sends a token (its object id) and
keeps the real ObjectRef; the worker holds an inert stand-in, and when the token
comes back the host substitutes its own ObjectRef again. An Opaque lists the
references inside it, so the PEs that carry it keep those alive.

The host keeps its ObjectRef while the worker may still use the token: it
counts the tokens it sends, and the worker, once it holds no stand-in for the
object any more, reports how many tokens that stand-in absorbed. The socket is
ordered, so the host drops its reference exactly when no token is in use or in
flight (see _refs.py for counting between PEs). Actor handles
coming from a worker are mapped to one host-side handle per worker-side handle,
so each keeps its own call order.

Every message between a host and a worker is two frames: a small header
``(kind, rid)``, pickled with plain pickle, and the payload. A payload that
cannot be deserialized fails only the request its header names."""

import collections
import io
import pickle
import threading
import weakref

from .object_ref import ObjectRef

PROTOCOL = pickle.HIGHEST_PROTOCOL


class Opaque(object):
    """A value serialized by a worker process (``data``), with the references
    it contains (``refs``: ObjectRefs in a Charm4py process, stand-ins in the
    worker that made it, which keep its tokens in use until it is sent)."""

    __slots__ = ('data', 'refs')

    def __init__(self, data, refs=()):
        self.data = data
        self.refs = list(refs)

    def __reduce__(self):
        return (Opaque, (self.data, self.refs))

    def __sizeof__(self):
        # the object store keeps small values at their home PE (sys.getsizeof)
        return len(self.data) + 64

    def load(self):
        """The value, for code that runs in a Charm4py process (isolation
        'none')."""
        return _PEUnpickler(io.BytesIO(self.data),
                            {ref.store_id: ref for ref in self.refs}).load()


def load_value(value):
    return value.load() if type(value) is Opaque else value


def encode_header(kind, rid=None):
    return pickle.dumps((kind, rid), PROTOCOL)


def decode_header(data):
    return pickle.loads(data)


# ---------------------------------------------------------------- host side

class _HostPickler(pickle.Pickler):

    def __init__(self, buf, side):
        super().__init__(buf, PROTOCOL)
        self.side = side

    def persistent_id(self, o):
        t = type(o)
        if t is ObjectRef:
            self.side.hand_out(o)
            return ('ref', o.store_id)
        if t is Opaque:
            for ref in o.refs:
                self.side.hand_out(ref)
            return ('opaque', o.data, [ref.store_id for ref in o.refs])
        return None


class _HostUnpickler(pickle.Unpickler):

    def __init__(self, buf, side):
        super().__init__(buf)
        self.side = side

    def persistent_load(self, pid):
        kind = pid[0]
        if kind == 'ref':
            return self.side.refs[pid[1]]
        if kind == 'opaque':
            return Opaque(pid[1], [self.side.refs[i] for i in pid[2]])
        if kind == 'actor':
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
        self.sent = collections.Counter()   # object id -> tokens sent, not released
        self.handles = {}    # worker handle id -> host ActorHandle

    def hand_out(self, ref):
        self.refs[ref.store_id] = ref
        self.sent[ref.store_id] += 1

    def release(self, released):
        """The worker gave up tokens: [(object id, how many)]."""
        for obj_id, count in released:
            left = self.sent[obj_id] - count
            if left > 0:
                self.sent[obj_id] = left
            else:
                del self.sent[obj_id]
                self.refs.pop(obj_id, None)

    def dumps(self, obj):
        buf = io.BytesIO()
        _HostPickler(buf, self).dump(obj)
        return buf.getvalue()

    def loads(self, data):
        return _HostUnpickler(io.BytesIO(data), self).load()


class _PEUnpickler(pickle.Unpickler):
    """Loads an Opaque in a Charm4py process."""

    def __init__(self, buf, refs):
        super().__init__(buf)
        self.refs = refs

    def persistent_load(self, pid):
        if pid[0] == 'ref':
            return self.refs[pid[1]]
        if pid[0] == 'actor':
            from .actor import ActorHandle
            return ActorHandle._from_state(pid[2])
        raise pickle.UnpicklingError('unknown persistent id ' + repr(pid))


# ---------------------------------------------------------------- worker side

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
        self.refs = {}

    def persistent_id(self, o):
        t = type(o)
        if t is ObjectRef:
            self.refs[o.store_id] = o
            return ('ref', o.store_id)
        if t is Opaque:
            # made by WorkerSide.opaque: its references are tokens the host knows
            return ('opaque', o.data, [ref.store_id for ref in o.refs])
        from .actor import ActorHandle
        if t is ActorHandle:
            return ('actor', o._uid, o._state())
        return None


class _WorkerUnpickler(pickle.Unpickler):

    def __init__(self, buf, side):
        super().__init__(buf)
        self.side = side

    def persistent_load(self, pid):
        kind = pid[0]
        if kind == 'ref':
            return self.side.token(pid[1])
        if kind == 'opaque':
            return self.side.loads(pid[1])
        if kind == 'actor':
            from .actor import ActorHandle
            return ActorHandle._from_state(pid[2])
        raise pickle.UnpicklingError('unknown persistent id ' + repr(pid))


class WorkerSide(object):
    """Translation in the worker process."""

    def __init__(self):
        # object id -> the live stand-in for it (one at a time, so references to
        # the same object are the same ObjectRef)
        self.refs = weakref.WeakValueDictionary()
        self.lock = threading.Lock()
        self.released = []   # (object id, tokens) for the host, see _Channel

    def token(self, obj_id):
        """The stand-in for a token that just arrived."""
        with self.lock:
            ref = self.refs.get(obj_id)
            if ref is None:
                ref = stand_in_ref(obj_id)
                # how many tokens this stand-in absorbed: reported when it dies
                ref._tokens = [0]
                weakref.finalize(ref, self._gone, obj_id, ref._tokens)
                self.refs[obj_id] = ref
            ref._tokens[0] += 1
            return ref

    def _gone(self, obj_id, tokens):
        self.released.append((obj_id, tokens))

    def take_released(self):
        """[(object id, tokens)] given up since the last call."""
        released, self.released = self.released, []
        return [(obj_id, cell[0]) for obj_id, cell in released]

    def dumps(self, obj):
        buf = io.BytesIO()
        _WorkerPickler(buf).dump(obj)
        return buf.getvalue()

    def loads(self, data):
        return _WorkerUnpickler(io.BytesIO(data), self).load()

    def opaque(self, value):
        """``value`` serialized for the host, which keeps it as an Opaque."""
        buf = io.BytesIO()
        pickler = _WorkerPickler(buf)
        pickler.dump(value)
        return Opaque(buf.getvalue(), list(pickler.refs.values()))

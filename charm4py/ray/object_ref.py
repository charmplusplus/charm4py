"""ObjectRef: the Ray-facing reference type.

An ObjectRef is a Charm4py object-store future (``Future(store=True)``), so the
existing object store and dependency buffering keep working. What this class
adds is Ray's identity semantics -- two references to the same object compare
and hash equal, and they print their ID the way ``ray.ObjectRef`` does -- and
reference counting: the object is freed once no ObjectRef for it exists in any
process (see _refs.py)."""

from ..threads import Future
from . import _refs

# True in worker processes, where references are inert stand-ins (see _ipc.py)
# that are neither counted nor pickled through Charm4py.
INERT = False


def _my_pe():
    import charm4py
    return charm4py.charm.myPe()


class ObjectRef(Future):

    def __init__(self, fid, gr, src, num_vals, store=True):
        # a reference does not need the greenlet that made it
        super().__init__(fid, None, src, num_vals, store=True)
        self._counted = not INERT
        if self._counted:
            _refs.created(self.store_id)

    def hex(self):
        # Ray object IDs are 28 bytes, printed as 56 hex digits
        return format(self.store_id, '056x')

    def binary(self):
        return bytes.fromhex(self.hex())

    def __eq__(self, other):
        return isinstance(other, ObjectRef) and other.store_id == self.store_id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.store_id)

    def __repr__(self):
        return 'ObjectRef(' + self.hex() + ')'

    __str__ = __repr__

    def __getstate__(self):
        if INERT:
            raise TypeError('an ObjectRef in a worker process is pickled by its '
                            'channel (see charm4py/ray/_ipc.py)')
        _refs.sent(self.store_id)
        return (self.store_id, _my_pe())

    def __setstate__(self, state):
        store_id, sender = state
        self.store_id = store_id
        self.fid = store_id & 0xffffffff
        self.src = store_id >> 32
        self.store = True
        self.parent = None
        self.num_borrowers = 0
        self.borrow_depth = 0
        self._requested = False
        self.gr = None
        self.nvals = 1
        self.values = []
        self.blocked = False
        self.gotvalues = False
        self.error = None
        self._counted = not INERT
        if self._counted:
            _refs.received(store_id, sender)

    def __del__(self):
        if self.__dict__.get('_counted'):
            try:
                _refs.dropped(self.store_id)
            except Exception:
                pass   # interpreter shutdown


class _NoneValue(object):
    """Stored in place of ``None``: the object store treats a missing object and a
    stored ``None`` the same way, so ``None`` cannot be stored directly."""

    def __reduce__(self):
        return (_NoneValue, ())


class _ErrorValue(object):
    """Stored in place of a value when the task that was to produce it failed.
    ``ray.get`` raises ``error`` instead of returning it, which keeps a task that
    legitimately returns an exception object distinguishable from a failed one."""

    def __init__(self, error):
        self.error = error

    def __reduce__(self):
        return (_ErrorValue, (self.error,))


NONE_VALUE = _NoneValue()


def wrap_value(value):
    if value is None:
        return NONE_VALUE
    return value


def unwrap_value(value):
    """Returns the stored value, raising the stored error if there is one. A
    value serialized by a worker process (an Opaque) is deserialized."""
    from ._ipc import Opaque
    if type(value) is Opaque:
        value = value.load()
    if isinstance(value, _NoneValue):
        return None
    if isinstance(value, _ErrorValue):
        error = value.error
        raise_stored_error(error.load() if type(error) is Opaque else error)
    return value


def raise_stored_error(error):
    from .exceptions import RayTaskError
    if isinstance(error, RayTaskError):
        raise error.as_instanceof_cause()
    raise error

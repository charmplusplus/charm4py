"""ObjectRef: the Ray-facing reference type.

An ObjectRef is a Charm4py object-store future (``Future(store=True)``), so the
existing object store, dependency buffering and borrower tracking all keep
working. What this class adds is Ray's identity semantics: two references to the
same object compare and hash equal, and they print and serialize their ID the way
``ray.ObjectRef`` does."""

from ..threads import Future

# True in actor worker processes, where references are inert stand-ins (see
# _ipc.py) that must never run Charm4py's pickling or deletion protocol.
INERT = False


class ObjectRef(Future):

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

    def __del__(self):
        if not INERT:
            Future.__del__(self)

    def __setstate__(self, state):
        # Future pickles through __getstate__/__setstate__, which does the
        # borrower bookkeeping; also give the copy the attributes a Future has
        Future.__setstate__(self, state)
        self.gr = None
        self.nvals = 1
        self.values = []
        self.blocked = False
        self.gotvalues = False
        self.error = None


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
    """Returns the stored value, raising the stored error if there is one."""
    if isinstance(value, _NoneValue):
        return None
    if isinstance(value, _ErrorValue):
        raise_stored_error(value.error)
    return value


def raise_stored_error(error):
    from .exceptions import RayTaskError
    if isinstance(error, RayTaskError):
        raise error.as_instanceof_cause()
    raise error

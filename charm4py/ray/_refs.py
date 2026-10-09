"""Reference counting for ObjectRefs: an object is freed once no process holds
a reference to it.

Charm++ does not guarantee message order, so this counting never relies on it.
Between PEs it is a borrower tree. Each PE counts, per object, its live
ObjectRef instances and the copies it has sent that are not yet accounted for
(``children``). A PE that receives a copy and holds none yet makes the sender its
*parent*: the sender keeps counting that copy as a child until this PE lets go
of all of its own. A PE that already holds the object tells the sender at once
that the copy is accounted for (its own parent chain keeps the object alive).
When a PE's count reaches zero it releases its parent; when the owner's (the
PE that created the reference, the root) reaches zero, the object is freed
everywhere. A copy in flight is always counted by its sender, so no count can
reach zero early whatever order messages arrive in.

Worker processes (see _ipc.py) hold tokens instead: their host keeps the real
ObjectRef while a worker may still use the token, and counts how often it sent
the token; the worker reports how many it received once it holds no stand-in
for the object any more. The socket is ordered, so the counts are exact.

Releases are batched and sent from the scheduler (``__del__`` can run anywhere,
including inside Charm4py's own code)."""

import collections

_live = collections.Counter()       # object id -> live ObjectRef instances here
_children = collections.Counter()   # object id -> copies sent and not accounted for
_parent = {}                        # object id -> PE this one got the object from

_pending = collections.defaultdict(list)   # PE -> object ids to release there
_flush_scheduled = False
enabled = True


def _charm():
    import charm4py
    return charm4py.charm


def created(obj_id):
    """A new reference, made by its owner (this PE)."""
    _live[obj_id] += 1


def sent(obj_id):
    """A copy of a reference is being pickled to be sent somewhere."""
    _children[obj_id] += 1


def received(obj_id, sender):
    """A copy sent by PE ``sender`` was unpickled here."""
    me = _charm().myPe()
    held = _live[obj_id] + _children[obj_id] > 0
    _live[obj_id] += 1
    if held or (obj_id >> 32 == me and sender != me and obj_id not in _parent):
        # accounted for already: by this PE's own hold, or because this PE is
        # the owner (the root needs no parent)
        _account(sender, obj_id)
    else:
        _parent[obj_id] = sender


def dropped(obj_id):
    """A live reference here was deleted."""
    _live[obj_id] -= 1
    _maybe_release(obj_id)


def released(obj_ids):
    """Copies this PE sent are accounted for (a release message)."""
    for obj_id in obj_ids:
        _children[obj_id] -= 1
        _maybe_release(obj_id)


def _account(pe, obj_id):
    if pe == _charm().myPe():
        _children[obj_id] -= 1
        _maybe_release(obj_id)
    else:
        _send_later(pe, obj_id)


def _maybe_release(obj_id):
    if _live[obj_id] > 0 or _children[obj_id] > 0:
        return
    _live.pop(obj_id, None)
    _children.pop(obj_id, None)
    parent = _parent.pop(obj_id, None)
    if parent is not None:
        _account(parent, obj_id)
    elif obj_id >> 32 == _charm().myPe():
        _free_later(obj_id)


_to_free = []


def _free_later(obj_id):
    _to_free.append(obj_id)
    _schedule_flush()


def _send_later(pe, obj_id):
    _pending[pe].append(obj_id)
    _schedule_flush()


def _schedule_flush():
    global _flush_scheduled
    if _flush_scheduled:
        return
    _flush_scheduled = True
    _charm().scheduleCallableAfter(_flush, 0)


def _flush():
    global _flush_scheduled, _pending, _to_free
    _flush_scheduled = False
    from . import api
    pending, _pending = _pending, collections.defaultdict(list)
    to_free, _to_free = _to_free, []
    if api._workers is None:
        return
    for pe, obj_ids in pending.items():
        api._workers[pe].release_refs(obj_ids)
    if to_free:
        store = api.object_store[_charm().myPe()].ckLocalBranch()
        for obj_id in to_free:
            store.free_object(obj_id)


def holds(obj_id):
    return _live[obj_id] + _children[obj_id] > 0

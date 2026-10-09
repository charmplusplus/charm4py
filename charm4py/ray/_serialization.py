"""How remote functions and actor classes travel to the PE that runs them.

``@ray.remote`` replaces a module-level name with a wrapper, so pickle can no
longer serialize the original function or class by reference (the name now
resolves to the wrapper). Instead a callable is described by its module and
qualified name, and the receiving PE looks that name up and unwraps it. Callables
that cannot be found by name (closures, lambdas, classes defined in a function)
are pickled by value with cloudpickle or dill when one of them is installed."""

import hashlib
import importlib
import sys

_resolved = {}


def _lookup(module, qualname):
    mod = sys.modules.get(module)
    if mod is None:
        mod = importlib.import_module(module)
    obj = mod
    for part in qualname.split('.'):
        obj = getattr(obj, part)
    return obj


def _unwrap(obj):
    return getattr(obj, '__ray_target__', obj)


def describe(obj, wrapper=None):
    module = getattr(obj, '__module__', None)
    qualname = getattr(obj, '__qualname__', None)
    if module and qualname and '<locals>' not in qualname and '<lambda>' not in qualname:
        try:
            found = _lookup(module, qualname)
        except Exception:
            found = None
        if found is not None and (found is obj or (wrapper is not None and found is wrapper)):
            return ('ref', module, qualname)
    for lib_name in ('cloudpickle', 'dill'):
        try:
            lib = importlib.import_module(lib_name)
        except ImportError:
            continue
        data = lib.dumps(obj)
        return ('value', lib_name, data, hashlib.sha1(data).hexdigest())
    raise TypeError('charm4py.ray cannot send ' + repr(obj) + ' to other processes: it '
                    'is not reachable by name from its module, and neither cloudpickle '
                    'nor dill is installed to serialize it by value')


def resolve(desc):
    key = desc if desc[0] == 'ref' else desc[3]
    obj = _resolved.get(key)
    if obj is None:
        if desc[0] == 'ref':
            obj = _unwrap(_lookup(desc[1], desc[2]))
        else:
            obj = _unwrap(importlib.import_module(desc[1]).loads(desc[2]))
        _resolved[key] = obj
    return obj

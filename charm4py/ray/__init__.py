"""The Ray Core API, implemented on Charm4py.

``from charm4py import ray`` gives the API inside a Charm4py program. To run
unmodified code that does ``import ray``, launch it with
``python -m charmrun.start +pN -m charm4py.ray.launch script.py``, which installs
this package as ``ray`` (see charm4py/ray/launch.py)."""

from . import exceptions
from .object_ref import ObjectRef
from .api import (init, shutdown, is_initialized, get, wait, put, kill, get_actor, cancel,
                  nodes, cluster_resources, available_resources, get_object_store)
from .runtime_context import get_runtime_context
from .remote_function import RemoteFunction
from .actor import ActorClass, ActorHandle, method, exit_actor
from . import actor, types, util, runtime_context, remote_function, _private
from ._private import state as _state  # noqa: F401  (import ray._private.state works)

# The Ray release whose API this implementation follows (see docs and tests).
__version__ = '2.56.1'
__charm4py__ = True

_ObjectRef = ObjectRef
ObjectID = ObjectRef


def remote(*args, **kwargs):
    """``@ray.remote`` / ``@ray.remote(**options)`` for functions and classes."""
    if len(args) == 1 and not kwargs and callable(args[0]):
        return _make_remote(args[0], {})
    if args:
        raise TypeError('The @ray.remote decorator must be applied either with no '
                        'arguments and no parentheses, for example `@ray.remote`, or it '
                        'must be applied using some of the arguments in the list ' +
                        str(sorted(set(_TASK_OPTIONS) | set(_ACTOR_OPTIONS))) +
                        ', for example `@ray.remote(num_returns=2, '
                        'resources={"CustomResource": 1})`.')

    def decorator(function_or_class):
        return _make_remote(function_or_class, kwargs)
    return decorator


def _make_remote(function_or_class, options):
    import inspect
    if inspect.isclass(function_or_class):
        return ActorClass(function_or_class, options)
    if callable(function_or_class):
        return RemoteFunction(function_or_class, options)
    raise TypeError('The @ray.remote decorator must be applied to either a function or '
                    'a class.')


from ._options import TASK_OPTIONS as _TASK_OPTIONS, ACTOR_OPTIONS as _ACTOR_OPTIONS  # noqa: E402

__all__ = ['init', 'shutdown', 'is_initialized', 'get', 'wait', 'put', 'kill', 'get_actor',
           'cancel', 'nodes', 'cluster_resources', 'available_resources', 'remote',
           'method', 'get_runtime_context', 'ObjectRef', 'exceptions', 'actor', 'types',
           'util']

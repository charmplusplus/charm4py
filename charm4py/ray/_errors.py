"""Building the errors stored when user code fails. Used both by the Charm hosts
and by actor worker processes, so this module must not import Charm4py chares."""

import os
import pickle
import traceback

from . import api
from .exceptions import RayTaskError, RaySystemError

_CHARM4PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def user_traceback(exc):
    """The traceback of ``exc`` without the frames of charm4py itself."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__)
              if not os.path.abspath(f.filename).startswith(_CHARM4PY_DIR)]
    lines = traceback.format_list(frames)
    lines += traceback.format_exception_only(type(exc), exc)
    return ''.join(lines)


def portable_cause(exc):
    """``exc`` if it survives pickling (it is stored in the object store and may be
    sent to other processes), else None."""
    try:
        pickle.loads(pickle.dumps(exc))
        return exc
    except Exception:
        return None


def task_error(function_name, exc, actor_repr=None, actor_id=None):
    tb = user_traceback(exc)
    cause = portable_cause(exc)
    if cause is None:
        cause = RaySystemError(tb)
    return RayTaskError(function_name, tb, cause, ip=api.get_node_ip_address(),
                        actor_repr=actor_repr, actor_id=actor_id)


def crash_error(function_name, code):
    return RayTaskError(function_name, '  SystemExit: ' + str(code) +
                        '\nThe task called sys.exit()', RaySystemError(
                            'The worker exited while executing this task (sys.exit(' +
                            str(code) + '))'), ip=api.get_node_ip_address())

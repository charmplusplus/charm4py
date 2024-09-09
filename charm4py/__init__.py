import sys
if sys.version_info < (2, 7, 0):
    raise RuntimeError('Charm4py requires Python 2.7 or higher')
import atexit
import os
try:
    import greenlet
except ImportError:
    print('Charm4py requires the greenlet package. It can be installed via pip')
    exit(-1)


charm4py_version = 'unknown'
try:
    from ._version import version as charm4py_version
except:
    try:
        import subprocess
        charm4py_version = subprocess.check_output(['git', 'describe'],
                                 cwd=os.path.dirname(__file__)).rstrip().decode()
    except:
        pass

if os.environ.get('CHARM_NOLOAD', '0') == '0':
    from .charm import register, charm, readonlies, Options
    Reducer = charm.reducers
    Future = charm.createFuture

    from .entry_method import when, coro, coro_ext, coro as threaded

    from .chare import Chare, Group, Array, ArrayMap
    from .channel import Channel
    from .object_store import ObjectStore

    def checkCharmStarted():
        if not charm.started:
            print('Program is exiting but charm was not started: charm.start() was not '
                  'called or error happened before start')

    atexit.register(checkCharmStarted)

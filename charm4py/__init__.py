import sys
if sys.version_info < (2, 7, 0):
    raise RuntimeError("charm4py requires Python 2.7 or higher")
import atexit
import os
import subprocess


charm4py_version = "unknown"
try:
    from ._version import version as charm4py_version
except:
    try:
        charm4py_version = subprocess.check_output(['git', 'describe'],
                                 cwd=os.path.dirname(__file__)).rstrip().decode()
    except:
        pass

if os.environ.get('CHARM_NOLOAD', '0') == '0':
    from .charm import charm, readonlies, Options
    Reducer = charm.reducers

    CkMyPe = charm.myPe
    CkNumPes = charm.numPes
    CkExit = charm.exit
    CkAbort = charm.abort

    from .entry_method import when, threaded

    from .chare import Chare, Group, Array, ArrayMap


    def checkCharmStarted():
        if not charm.started:
            print('Program is exiting but charm was not started: charm.start() was not '
                  'called or error happened before start')


    atexit.register(checkCharmStarted)

import sys
if sys.version_info < (2, 7, 0):
    raise RuntimeError("charmpy requires Python 2.7 or higher")
import atexit


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

"""
charm.options.cprofiling: every PE runs under cProfile and dumps its statistics
at process exit to <charm.options.cprofile_filename><pe>.prof.

Run (the auto_test form; the files go to <tempdir>/charm4py-cprofile/):

    python -m charmrun.start tests/thin/cprofile.py +p2

With the argument `default` the program keeps the default file name ("profile",
so profile0.prof, profile1.prof in the current directory); with `dir=DIR` the
files go to DIR. The program calls cprofile_marker_work on every PE
and ends with charm.exit() from a plain (not coroutine) entry method.

Check afterwards (not part of the auto_test run, which cannot look at files after
the processes end):

    python tests/thin/cprofile.py check DIR NPES

which loads DIR/profile<pe>.prof for every PE with pstats.Stats and requires an
entry for cprofile_marker_work in each.
"""
import os
import sys
import tempfile

if len(sys.argv) >= 2 and sys.argv[1] == 'check':
    import pstats
    d, n = sys.argv[2], int(sys.argv[3])
    for pe in range(n):
        path = os.path.join(d, 'profile%d.prof' % pe)
        if not os.path.isfile(path):
            print('cprofile: FAIL, no file', path)
            sys.exit(1)
        st = pstats.Stats(path)
        calls = [v[1] for k, v in st.stats.items() if k[2] == 'cprofile_marker_work']
        if not calls:
            print('cprofile: FAIL,', path, 'has no entry for cprofile_marker_work')
            sys.exit(1)
        print('cprofile: PE %d: %s loads, %d functions, cprofile_marker_work called %d times' %
              (pe, path, len(st.stats), calls[0]))
    print('cprofile: PASS, profiles of all %d PEs load and contain the program\'s function' % n)
    sys.exit(0)

from charm4py import charm, Chare, Group  # noqa: E402

charm.options.cprofiling = True
if 'default' not in sys.argv:
    d = os.path.join(tempfile.gettempdir(), 'charm4py-cprofile')
    for a in sys.argv[1:]:
        if a.startswith('dir='):
            d = a[4:]
    os.makedirs(d, exist_ok=True)
    charm.options.cprofile_filename = os.path.join(d, 'profile')


def cprofile_marker_work(n):
    return sum(i * i for i in range(n))


class Worker(Chare):

    def work(self, main):
        for _ in range(10):
            cprofile_marker_work(1000)
        main.done()


class Main(Chare):

    def __init__(self, args):
        self.n = 0
        Group(Worker).work(self.thisProxy)

    def done(self):
        self.n += 1
        if self.n == charm.numPes():
            print('cprofile: profiles go to', charm.options.cprofile_filename + '<pe>.prof')
            charm.exit()


charm.start(Main)

"""A failure inside the runtime's startup reaches Python as an exception.

Run: python -m charmrun.start tests/thin/startup_error.py +p1   (and +p2)

The parent starts a child run as a single process with FI_PROVIDER naming no
libfabric provider, so on a libfabric-only LCI the child's CkDynStart finds no
network backend. LCI reports that by throwing std::runtime_error. charm.start
releases the GIL around CkDynStart, and the exception used to unwind past the
call that took the GIL back, so nanobind set the Python error without a thread
state and the child died with SIGSEGV before printing anything (seen on the
macOS CI run of #326). The child must end with an exit code, not a signal:
a RuntimeError (exit 1), or a normal start where another network backend works.
"""
import os
import subprocess
import sys
from charm4py import charm

CHILD = '--child' in sys.argv


def main(args):
    if CHILD:
        print('STARTUP-ERROR CHILD STARTED')
        charm.exit()
        return
    # a single-process child: drop the launcher's PMI and Slurm variables, and
    # LCT_ABORT_ON_ASSERT, which would make LCI abort instead of throwing
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('LCT_', 'LCI_', 'PMI_', 'PMIX_', 'SLURM_', 'FI_'))}
    env['FI_PROVIDER'] = 'nosuchprovider'
    p = subprocess.run([sys.executable, os.path.abspath(__file__), '--child', '+pe', '1'],
                       env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       timeout=120)
    out = p.stdout
    started = 'STARTUP-ERROR CHILD STARTED' in out
    raised = 'RuntimeError' in out
    if p.returncode < 0 or not (started or raised):
        charm.abort('FAIL startup_error: the child run with no network backend ended with %d '
                    '(a negative value is the signal); its output:\n%s' % (p.returncode, out[-2000:]))
    print('PASS startup_error: the child run with no network backend %s (exit %d)'
          % ('started on another backend' if started else 'raised RuntimeError', p.returncode))
    charm.exit()


charm.start(main)

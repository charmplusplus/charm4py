import sys
import os
import os.path


def executable_is_python(args):
    """
    Determines whether the first executable passed to args is a
    Python file. Other valid examples include analysis tools
    such as Perf that will run the actual Python program.

    Note: Returns true if no executable was found or if an executable
    was found and that executable is a Python file.
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    def is_pyfile(fpath):
        return os.path.isfile(fpath) and fpath.endswith(".py")
    for each in args:
        if is_pyfile(each):
            return True
        if is_exe(each):
            return False
    # No executable was found, but we'll let Python tell us
    return True


def nodelist_islocal(filename, regexp):
    if not os.path.exists(filename):
        # it is an error if filename doesn't exist, but I'll let charmrun print
        # the error. don't add ++local so that charmrun detects it
        return False
    with open(filename, 'r') as f:
        for line in f:
            m = regexp.search(line)
            if m is not None and m.group(1) not in {'localhost', '127.0.0.1'}:
                return False
    return True


def checkNodeListLocal(args):
    import re
    regexp = re.compile(r"^\s*host\s+(\S+)\s*$")

    try:
        i = args.index('++nodelist')
    except ValueError:
        i = -1
    if i != -1:
        return nodelist_islocal(args[i+1], regexp)

    if 'NODELIST' in os.environ:
        return nodelist_islocal(os.environ['NODELIST'], regexp)

    nodelist_cur_dir = os.path.join(os.getcwd(), 'nodelist')
    if os.path.exists(nodelist_cur_dir):
        return nodelist_islocal(nodelist_cur_dir, regexp)

    nodelist_home_dir = os.path.join(os.path.expanduser('~'), '.nodelist')
    if os.path.exists(nodelist_home_dir):
        return nodelist_islocal(nodelist_home_dir, regexp)

    return True


def reconverse_runtime_is_installed():
    package_root = os.path.dirname(os.path.dirname(__file__))
    library_dir = os.path.join(package_root, 'charm4py', '.libs')
    return any(os.path.isfile(os.path.join(library_dir, filename))
               for filename in ('reconverse',
                                'libcharm4py_reconverse.dylib',
                                'libcharm4py_reconverse.so'))


def reconverse_args(args):
    """Translate Charm++'s +p option to lcrun processes and +pe PEs."""
    translated = []
    num_pes = 1
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in ('+p', '+pe'):
            if index + 1 >= len(args):
                raise ValueError(arg + ' requires a process count')
            num_pes = int(args[index + 1])
            index += 2
        elif arg.startswith('+pe') and arg[3:].isdigit():
            num_pes = int(arg[3:])
            index += 1
        elif arg.startswith('+p') and arg[2:].isdigit():
            num_pes = int(arg[2:])
            index += 1
        elif arg == '++local':
            index += 1
        else:
            translated.append(arg)
            index += 1
    translated.extend(['+pe', str(num_pes)])
    return num_pes, translated


def prepare_file_pmi_dir():
    """Creates the shared rendezvous directory that LCI's file PMI uses when it
    does not know LCT_PMI_FILE_DIR (LCI before uiuc-hpc/lci#203):
    ~/.tmp/lct_pmi_file-<SLURM_JOBID, or 0>/. Every rank creates it when it is
    missing, and when the ranks of a job race to create it the job can fail to
    start or hang (3 of 10 launches on a fresh directory, none of 30 with it
    created first)."""
    import re
    jobid = re.match(r'\d*', os.environ.get('SLURM_JOBID', '')).group()
    home = os.environ.get('HOME', os.path.expanduser('~'))
    path = os.path.join(home, '.tmp', 'lct_pmi_file-' + str(int(jobid or 0)))
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass   # the ranks will try, and report it


def launch_reconverse(num_pes, args, grace=5.0):
    """Runs a Reconverse job of ``num_pes`` processes on this machine (see
    _launch_reconverse), with a file-PMI rendezvous directory of its own."""
    import shutil
    import tempfile
    pmi_dir = tempfile.mkdtemp(prefix='lct-pmi-file.')
    try:
        return _launch_reconverse(num_pes, args, grace, pmi_dir)
    finally:
        shutil.rmtree(pmi_dir, ignore_errors=True)


def _launch_reconverse(num_pes, args, grace, pmi_dir):
    """Starts the job's ``num_pes`` processes on this machine and waits for them.

    This does what LCI's ``lcrun`` does (the same bootstrap variables, one
    process per PE, and the rest of the job stopped when a process fails), so
    that a Charm4py installation can start jobs without lcrun, which is a
    script in LCI's source tree. Processes still running ``grace`` seconds
    after that SIGTERM are killed. Returns the first failing process's exit
    code (128 + N for signal N), or 0."""
    import signal
    import subprocess
    import time

    cmd = ([sys.executable] if executable_is_python(args) else []) + list(args)
    env = dict(os.environ)
    # as lcrun: LCI's PMI takes the rank count from the environment and its
    # peers from a file
    env['LCT_PMI_BACKEND'] = 'file,local'
    env['LCT_PMI_FILE_NRANKS'] = str(num_pes)
    # LCI meets in LCT_PMI_FILE_DIR when it is set (uiuc-hpc/lci#203), as
    # lcrun now gives each job; older LCI uses one shared directory per host
    env['LCT_PMI_FILE_DIR'] = pmi_dir
    if num_pes > 1:
        prepare_file_pmi_dir()

    # the job gets a process group of its own, and SIGINT and SIGTERM sent to
    # this process are passed on to all of it
    group = [0]

    def join_group():
        os.setpgid(0, group[0])

    procs = {}
    try:
        for _ in range(num_pes):
            proc = subprocess.Popen(cmd, env=env, preexec_fn=join_group)
            if not group[0]:
                group[0] = proc.pid
            try:
                os.setpgid(proc.pid, group[0])   # whichever of us runs first
            except OSError:
                pass
            procs[proc.pid] = proc
    except OSError as error:
        print('could not start', cmd[0] + ':', error)
        if group[0]:
            os.killpg(group[0], signal.SIGKILL)
        return 1

    def forward(signum, frame):
        try:
            os.killpg(group[0], signum)
        except OSError:
            pass
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, forward)

    def exit_code(status):
        if os.WIFSIGNALED(status):
            return 128 + os.WTERMSIG(status)
        return os.WEXITSTATUS(status)

    result = 0
    deadline = None      # set once the job is being stopped
    while procs:
        if deadline is None:
            pid, status = os.waitpid(-1, 0)
        else:
            pid, status = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                if time.monotonic() > deadline:
                    try:
                        os.killpg(group[0], signal.SIGKILL)
                    except OSError:
                        pass
                    deadline = float('inf')
                time.sleep(0.05)
                continue
        proc = procs.pop(pid, None)
        if proc is None:
            continue
        proc.returncode = code = exit_code(status)
        if code != 0 and result == 0:
            result = code
            if procs:
                print('charmrun: a process of the job exited with code %d; stopping the '
                      'other %d' % (code, len(procs)), file=sys.stderr, flush=True)
                try:
                    os.killpg(group[0], signal.SIGTERM)
                except OSError:
                    pass
                deadline = time.monotonic() + grace
    return result


def start(args=None):
    import subprocess

    if args is None or len(args) == 0:
        args = sys.argv[1:]
    else:
        args = list(args)

    if '++interactive' in args and 'charm4py.interactive' not in args:
        args += ['-m', 'charm4py.interactive']

    if reconverse_runtime_is_installed():
        try:
            num_pes, args = reconverse_args(args)
        except (ValueError, TypeError) as error:
            print('Invalid Reconverse process count:', error)
            return 1

        return launch_reconverse(num_pes, args)

    if '++local' not in args and '++mpiexec' not in args and checkNodeListLocal(args):
        args.append('++local')

    cmd = [os.path.join(os.path.dirname(__file__), 'charmrun')]
    if executable_is_python(args):
        # Note: sys.executable is the absolute path to the Python interpreter
        # We only want to invoke the interpreter if the execution target is a
        # Python file
        cmd.append(sys.executable)  # for example: /usr/bin/python3
    cmd.extend(args)
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print('charmrun executable not found. You are running \"' + __file__ + '\"')
        print('Make sure this is a built or installed version of charmrun')
        return 1


if __name__ == '__main__':
    sys.exit(start())

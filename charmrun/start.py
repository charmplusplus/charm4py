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
    regexp = re.compile("^\s*host\s+(\S+)\s*$")

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


def start(args=[]):
    import subprocess

    if len(args) == 0:
        args = sys.argv[1:]
    if '++local' not in args and '++mpiexec' not in args and checkNodeListLocal(args):
        args.append('++local')

    if '++interactive' in args and 'charm4py.interactive' not in args:
        args += ['-m', 'charm4py.interactive']

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

import sys
import os
import os.path
import shutil


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

        lcrun = os.environ.get('CHARM4PY_LCRUN', os.environ.get('LCRUN'))
        if lcrun is None:
            lcrun = shutil.which('lcrun')
        if lcrun is None:
            print('Reconverse requires lcrun. Set CHARM4PY_LCRUN or LCRUN '
                  'to the lcrun executable.')
            return 1

        cmd = [lcrun, '-n', str(num_pes)]
        if executable_is_python(args):
            cmd.append(sys.executable)
        cmd.extend(args)
        try:
            return subprocess.call(cmd)
        except FileNotFoundError:
            print('lcrun executable not found:', lcrun)
            return 1

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

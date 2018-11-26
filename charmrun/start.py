import sys
import os
import os.path


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

    try:
        idx = args.index('++interactive')
        args[idx] = '-c'
        if os.name == 'nt':
            # workaround for how windows charmrun executable passes argument
            args.insert(idx + 1, '\"from charm4py import charm ; charm.start(interactive=True)\"')
        else:
            args.insert(idx + 1, 'from charm4py import charm ; charm.start(interactive=True)')
    except ValueError:
        pass

    cmd = [os.path.join(os.path.dirname(__file__), 'charmrun')]
    cmd.append(sys.executable)
    cmd.extend(args)
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print('charmrun executable not found. You are running \"' + __file__ + '\"')
        print('Make sure this is a built or installed version of charmrun')
        return 1


if __name__ == '__main__':
    sys.exit(start())

import time
import subprocess
import sys
if sys.version_info[0] < 3:
    print("auto_test requires Python 3")
    exit(1)
import os
import shutil
from collections import defaultdict
import json


if len(sys.argv) == 2 and sys.argv[1] == '-version_check':
    exit(sys.version_info[0])


def searchForPython(python_implementations):
    py3_exec = None
    py3_exec = shutil.which('python3')
    if py3_exec is None:
        exec_str = shutil.which('python')
        if exec_str is not None:
            version = subprocess.call([exec_str, 'auto_test.py', '-version_check'])
            if version >= 3:
                py3_exec = exec_str
    if py3_exec is None:
        print("WARNING: Python 3 executable not found for auto_test. If desired, set manually")
    else:
        python_implementations.add((3, py3_exec))


# ----------------------------------------------------------------------------------
TIMEOUT = 60  # timeout for each test (in seconds)
CHARM_QUIET_AFTER_NUM_TESTS = 5

commonArgs = ['++local']
default_num_processes = int(os.environ.get('CHARM4PY_TEST_NUM_PROCESSES', 4))

try:
    import numba
    numbaInstalled = True
except:
    numbaInstalled = False

# search for python executables
python_implementations = set()   # python implementations can also be added here manually
searchForPython(python_implementations)

interfaces = ['cython']

with open('test_config.json', 'r') as infile:
    tests = json.load(infile)

num_tests = 0
durations = defaultdict(dict)
for test in tests:
    if 'condition' in test:
        if test['condition'] == 'numbaInstalled' and not numbaInstalled:
            continue
        if test['condition'] == 'not numbaInstalled' and numbaInstalled:
            continue
    num_processes = max(test.get('force_min_processes', default_num_processes), default_num_processes)
    for interface in interfaces:
        durations[interface][test['path']] = []
        for version, python in sorted(python_implementations):
            if version < test.get('requires_py_version', -1):
                continue
            additionalArgs = []
            if num_tests >= CHARM_QUIET_AFTER_NUM_TESTS and '++quiet' not in commonArgs:
                additionalArgs.append('++quiet')
            cmd = ['charmrun/charmrun']
            if test.get('prefix'):
                cmd += [test['prefix']]
            if not test.get('interactive', False):
                cmd += [python] + [test['path']]
            else:
                cmd += [python] + ['-m', 'charm4py.interactive']
            if 'args' in test:
                cmd += test['args'].split(' ')
            cmd += commonArgs
            cmd += ['+p' + str(num_processes), '+libcharm_interface', interface]
            cmd += additionalArgs
            print('Test command is ' + ' '.join(cmd))
            startTime = time.time()
            stdin = None
            if test.get('interactive', False):
                stdin = open(test['path'])
            p = subprocess.Popen(cmd, stdin=stdin)
            try:
                rc = p.wait(TIMEOUT)
            except subprocess.TimeoutExpired:
                print("Timeout (" + str(TIMEOUT) + " secs) expired when running " + test['path'] + ", Killing process")
                p.kill()
                rc = -1
            if rc != 0:
                print("ERROR running test " + test['path'] + " with " + python)
                exit(1)
            else:
                elapsed = round(time.time() - startTime, 3)
                durations[interface][test['path']].append(elapsed)
                print("\n\n--------------------- TEST PASSED (in " + str(elapsed) + " secs) ---------------------\n\n")
                num_tests += 1


print("ALL TESTS (" + str(num_tests) + ") PASSED")
print("Durations:")
for interface in interfaces:
    print("\n---", interface, "---")
    for test, results in sorted(durations[interface].items()):
        print(test + ": " + str(results))

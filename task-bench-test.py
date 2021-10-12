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

num_processes = 4
python = None
for version, py in sorted(python_implementations):
    if version >= 3:
        python = py
        break
print(python)
cmd = ['charmrun/charmrun', ]
cmd += [python]
cmd += ["task-bench/charm4py/task_bench.py"]
cmd += ['+p' + str(num_processes), '+libcharm_interface', 'cython']

print(cmd)
startTime = time.time()
stdin = None
p = subprocess.Popen(cmd, stdin=stdin)
try:
    rc = p.wait(TIMEOUT)
except subprocess.TimeoutExpired:
    print("Timeout (" + str(TIMEOUT) + " secs) expired when running  task-bench, Killing process")
    p.kill()
    rc = -1
if rc != 0:
    print("ERROR running test " + "task-bench" + " with " + python)
    exit(1)
else:
    elapsed = round(time.time() - startTime, 3)
    # durations[interface][test['path']].append(elapsed)
    print("\n\n--------------------- TEST PASSED (in " + str(elapsed) + " secs) ---------------------\n\n")
    # num_tests += 1

import time
import subprocess
import sys
if sys.version_info[0] < 3:
  print("auto_test requires Python 3")
  exit(1)
import shutil

if len(sys.argv) == 2 and sys.argv[1] == '-version_check':
  exit(sys.version_info[0])

def searchForPython(python_implementations):
  py2_exec, py3_exec = None, None
  py2_exec = shutil.which('python2')
  if py2_exec is None:
    exec_str = shutil.which('python')
    if exec_str is not None:
      version = subprocess.call([exec_str, 'auto_test.py', '-version_check'])
      if version < 3: py2_exec = exec_str
      else: py3_exec = exec_str
  if py2_exec is None:
    print("WARNING: Python 2 executable not found for auto_test. If desired, set manually")
  else:
    python_implementations.add(py2_exec)
  if py3_exec is None:
    py3_exec = shutil.which('python3')
  if py3_exec is None:
    print("WARNING: Python 3 executable not found for auto_test. If desired, set manually")
  else:
    python_implementations.add(py3_exec)

# ----------------------------------------------------------------------------------
TIMEOUT=60  # in seconds

tests = []
tests.append(['tests/when/when_test.py'])
tests.append(['tests/reductions/group_reduction.py'])
tests.append(['tests/reductions/array_reduction.py'])
tests.append(['tests/reductions/custom_reduction.py'])
tests.append(['tests/reductions/test_gather.py'])
tests.append(['tests/reductions/bench_reductions.py'])
tests.append(['tests/dcopy/test_dcopy.py'])
tests.append(['tests/element_proxy/array_element_proxy.py'])
tests.append(['tests/element_proxy/group_element_proxy.py'])
tests.append(['tests/collections/test.py'])
tests.append(['examples/hello/group_hello.py'])
tests.append(['examples/hello/group_hello2.py'])
tests.append(['examples/hello/array_hello.py'])
tests.append(['examples/hello/dynamic_array.py'])
tests.append(['examples/multi-module/main.py'])
tests.append(['examples/particle/particle.py'])
tests.append(['examples/stencil3d/stencil3d_numba.py', '64', '32'])
tests.append(['examples/wave2d/wave2d.py', '1500'])
tests.append(['examples/tutorial/start.py'])
tests.append(['examples/tutorial/reduction.py'])
tests.append(['examples/tutorial/hello_world.py'])

commonArgs = ['++local', '+p4']

# search for python executables
python_implementations = set()   # python implementations can also be added here manually
searchForPython(python_implementations)

durations = {}
for test in tests:
  durations[test[0]] = []
  for python in sorted(python_implementations):
    cmd = ['./charmrun']
    cmd += [python]
    cmd += test
    cmd += commonArgs
    print("Test command is " + str(cmd))
    startTime = time.time()
    p = subprocess.Popen(cmd)
    try:
      rc = p.wait(TIMEOUT)
    except subprocess.TimeoutExpired:
      print("Timeout (" + str(TIMEOUT) + " secs) expired when running " + str(test) + ", Killing process")
      p.kill()
      rc = -1
    if rc != 0:
      print("ERROR running test " + str(test) + " with " + python)
      exit(1)
    else:
      elapsed = round(time.time() - startTime,3)
      durations[test[0]].append(elapsed)
      print("\n\n--------------------- TEST PASSED (in " + str(elapsed) + " secs) ---------------------\n\n")

print("ALL TESTS PASSED")
print("Durations:")
for test,results in sorted(durations.items()): print(test + ": " + str(results))

import time
import subprocess
import sys
if sys.version_info[0] < 3:
  print("auto_test requires Python 3")
  exit(1)
import shutil
from collections import defaultdict

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
    python_implementations.add((2,py2_exec))
  if py3_exec is None:
    py3_exec = shutil.which('python3')
  if py3_exec is None:
    print("WARNING: Python 3 executable not found for auto_test. If desired, set manually")
  else:
    python_implementations.add((3,py3_exec))

# ----------------------------------------------------------------------------------
TIMEOUT=60  # in seconds

tests = []
tests.append(['tests/array_maps/test1.py'])
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
tests.append(['tests/trees/topo_treeAPI.py'])
tests.append(['tests/migration/test_migrate.py', '+balancer', 'GreedyRefineLB', '+LBDebug', '1'])
tests.append(['tests/migration/chare_migration.py'])
tests.append(['tests/thread_entry_methods/test1.py'])
tests.append(['tests/thread_entry_methods/test1_when.py'])
tests.append(['tests/thread_entry_methods/test_main.py'])
tests.append(['tests/thread_entry_methods/future_reduction.py'])
tests.append(['tests/thread_entry_methods/future_bcast.py'])
tests.append(['tests/futures/test_futures.py'])
tests.append(['tests/futures/multi_futures.py'])
tests.append(['examples/hello/group_hello.py'])
tests.append(['examples/hello/group_hello2.py'])
tests.append(['examples/hello/array_hello.py'])
tests.append(['examples/hello/dynamic_array.py'])
tests.append(['examples/hello/cons_args_hello.py'])
tests.append(['examples/multi-module/main.py'])
tests.append(['examples/particle/particle.py', '+balancer', 'GreedyRefineLB'])
tests.append(['examples/stencil3d/stencil3d_numba.py', '64', '32', '+balancer', 'GreedyRefineLB', '+LBDebug', '1'])
tests.append(['examples/wave2d/wave2d.py', '1500'])
tests.append(['examples/tutorial/start.py'])
tests.append(['examples/tutorial/chares.py'])
tests.append(['examples/tutorial/reduction.py'])
tests.append(['examples/tutorial/hello_world.py'])
tests.append(['examples/tutorial/hello_world2.py'])

commonArgs = ['++local', '+p4']

# search for python executables
python_implementations = set()   # python implementations can also be added here manually
searchForPython(python_implementations)

interfaces = ['ctypes', 'cffi', 'cython']
supported_py_versions = {'ctypes': {2, 3},
                         'cffi'  : {2, 3},
                         'cython': {3} }

durations = defaultdict(dict)
for interface in interfaces:
  for test in tests:
    durations[interface][test[0]] = []
    for version,python in sorted(python_implementations):
      if version not in supported_py_versions[interface]: continue
      cmd = ['./charmrun']
      cmd += [python]
      cmd += test
      cmd += commonArgs
      cmd += ['+libcharm_interface', interface]
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
        durations[interface][test[0]].append(elapsed)
        print("\n\n--------------------- TEST PASSED (in " + str(elapsed) + " secs) ---------------------\n\n")

print("ALL TESTS PASSED")
print("Durations:")
for interface in interfaces:
  print("\n---", interface, "---")
  for test,results in sorted(durations[interface].items()): print(test + ": " + str(results))

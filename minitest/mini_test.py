import time
import subprocess
import sys
if sys.version_info[0] < 3:
    print("mini_test requires Python 3")
    exit(1)
import os
import shutil
from collections import defaultdict
import json

def mini_test():
    if len(sys.argv) != 2:
        print("Usage: python mini_test.py [test_name: group_hello, array_hello, simple_ray]")
        exit(1)

    script_dir = os.path.dirname(os.path.realpath(__file__))

    test_name = sys.argv[1]
    test_file = script_dir + '/' + test_name + '.py'

    if not os.path.exists(test_file):
        print("Test " + test_file + " not found")
        exit(1)
        
    # TODO: add arguments for libcharm_interface



    commonArgs = ['++local']
    default_num_processes = int(os.environ.get('CHARM4PY_TEST_NUM_PROCESSES', 4))

    num_processes = default_num_processes
    python_executable = sys.executable
    TIMEOUT = 120  # timeout for each test (in seconds)

            
    cmd = [python_executable]
    cmd += ["-m", "charmrun.start"]

    cmd += commonArgs
    cmd += ['+p' + str(num_processes)]
    # TODO: optional interface setting
    # cmd += ['+libcharm_interface', interface]
    cmd += [test_file]
    print('Test command is ' + ' '.join(cmd))
    startTime = time.time()
    stdin = None

    p = subprocess.Popen(cmd, stdin=stdin)
    try:
        rc = p.wait(TIMEOUT)
    except subprocess.TimeoutExpired:
        print("Timeout (" + str(TIMEOUT) + " secs) expired when running " + test_name + ", Killing process")
        p.kill()
        rc = -1
    if rc != 0:
        print("ERROR running test " + test_name + " with " + python_executable)
        exit(1)
    else:
        elapsed = round(time.time() - startTime, 3)
        
        print("\n\n--------------------- TEST RAN (in " + str(elapsed) + " secs) ---------------------\n\n")
    
if __name__ == "__main__":
    mini_test()
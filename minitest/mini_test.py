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
import argparse

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run a minimal charm4py example."
    )
    
    # Add the required positional argument
    parser.add_argument(
        'example',
        type=str,
        choices=['group_hello', 'array_hello', 'simple_ray'],
        help='The name of the example to run (required). Current supported example group_hello, array_hello, simple_ray.'
    )
    
    # Add the optional argument
    parser.add_argument(
        '-i', '--interface',
        type=str,
        help='Specify the libcharm interface to use (optional). Current supported interfaces: ctypes, cython, cffi.'
    )
    
    parser.add_argument(
        '-n', '--num_processes',
        type=int,
        help='Specify the number of processes to use (optional). Default is 4.'
    )

    # Parse the arguments
    return parser.parse_args()
    
def mini_test():
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.realpath(__file__))

    example = args.example
    example_file = script_dir + '/' + example + '.py'

    if not os.path.exists(example_file):
        print("Example" + example_file + " not found")
        exit(1)
    

    TIMEOUT = 120  # timeout for each test (in seconds)
 
    cmd = ["charmrun"]
    cmd += ['++local']
    
    num_processes = 4
    if args.num_processes:
        num_processes = args.num_processes
        
    cmd += ['+p' + str(num_processes)]
    
    
   
    cmd += [example_file]
    
    if args.interface:
        cmd += ['+libcharm_interface', args.interface]
        
    print('Command is: ' + ' '.join(cmd))
    startTime = time.time()
    stdin = None

    p = subprocess.Popen(cmd, stdin=stdin)
    try:
        rc = p.wait(TIMEOUT)
    except subprocess.TimeoutExpired:
        print("Timeout (" + str(TIMEOUT) + " secs) expired when running " + example + ", Killing process")
        p.kill()
        rc = -1
    if rc != 0:
        print("ERROR running example " + example)
        exit(1)
    else:
        elapsed = round(time.time() - startTime, 3)
        
        print("\n\n--------------------- EXAMPLE RAN (in " + str(elapsed) + " secs) ---------------------\n\n")
    
if __name__ == "__main__":
    mini_test()
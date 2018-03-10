import sys
import os

def check_libcharm_version(lib):
    req_version = tuple([int(n) for n in open('charmpy/libcharm_version', 'r').read().split('.')])
    commit_id_str = ctypes.c_char_p.in_dll(lib, "CmiCommitID").value.decode()
    version = [int(n) for n in commit_id_str.split('-')[0][1:].split('.')]
    version = tuple(version + [int(commit_id_str.split('-')[1])])
    if version < req_version:
      req_str = '.'.join([str(n) for n in req_version])
      cur_str = '.'.join([str(n) for n in version])
      print("Charm++ version >= " + req_str + " required. Existing version is " + cur_str)
      exit(1)


if len(sys.argv) < 2:
  print("Usage: python setup.py CHARM_PATH")
  exit(1)
else:
  libcharmPath = sys.argv[1] + "/lib"
  charm_include_path = sys.argv[1] + "/include"
  if not os.path.isfile(charm_include_path + "/charm.h"):
    print("charm.h not found in " + charm_include_path)
    exit(1)
  try:
    import ctypes
    libcharm = ctypes.CDLL(libcharmPath + "/libcharm.so")
    check_libcharm_version(libcharm)
    del libcharm
    del ctypes
  except:
    print("Error accessing " + libcharmPath + "/libcharm.so " +
          "(library not found or is not valid)")
    raise sys.exc_info()[1]

cffiFound = False
if '--with-ctypes' not in sys.argv:
  try:
    import cffi
    version = tuple(int(v) for v in cffi.__version__.split('.'))
    if version[0] > 1: cffiFound = True
    elif version[0] == 1:
      if version[1] >= 7: cffiFound = True
    if not cffiFound:
      print("charmpy requires cffi >= 1.7. Installed version is " + cffi.__version__)
  except ImportError:
    print("cffi is not installed")

if not cffiFound:
  print("Will use ctypes interface. NOTE: cffi is highly recommended for best performance")
  libcharm_interface = 'ctypes'
else:
  libcharm_interface = 'cffi'
  import subprocess
  cmd = [sys.executable, 'charmlib_cffi_build.py', charm_include_path, libcharmPath]
  p = subprocess.Popen(cmd, shell=False, cwd=os.getcwd() + '/charmpy')
  rc = p.wait()
  print('/////////////////////////////')
  if rc == 0:
    print("\ncffi libcharm wrapper compiled successfully")
  else:
    print("\nERROR: cffi libcharm wrapper did NOT compile successfully")
    exit(1)

# generate charmpy config file
import json
config = {}
config['libcharm_interface'] = libcharm_interface
config['libcharm_path'] = libcharmPath
json.dump(config, open('charmpy/charmpy.cfg','w'))
print('charmpy.cfg file generated')

print('Setup complete\n')

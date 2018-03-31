import sys
import os

# -------------------------------------------------------------------------
# First step: check CHARM_PATH for existence of 'include/charm.h' and valid
# 'lib/libcharm.so'

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
    libcharmPath       = sys.argv[1] + "/lib"
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

# -------------------------------------------------------------------------
# Next choose and setup best way to interface with libcharm (ctypes, cffi or cython)

def check_cython():
    if sys.version_info[0] < 3:
        print("Charmpy cython layer not supported with Python 2")
    else:
        try:
            import cython
            return True
        except ImportError:
            print("cython is not installed")
    return False

def check_cffi():
    try:
        import cffi
        version = tuple(int(v) for v in cffi.__version__.split('.'))
        if version[0] > 1: return True
        elif version[0] == 1:
            if version[1] >= 7: return True
        print("charmpy requires cffi >= 1.7. Installed version is " + cffi.__version__)
    except ImportError:
        print("cffi is not installed")
    return False

def use_ctypes():
    print("Will use ctypes interface layer. NOTE: cython is highly recommended for best performance")
    global libcharm_interface
    libcharm_interface = 'ctypes'

def use_cffi():
    print("Will use cffi interface")
    import subprocess
    global libcharm_interface
    libcharm_interface = 'cffi'
    cmd = [sys.executable, 'charmlib_cffi_build.py', charm_include_path, libcharmPath]
    p = subprocess.Popen(cmd, shell=False, cwd=os.getcwd() + '/charmpy')
    rc = p.wait()
    print('/////////////////////////////')
    if rc == 0:
        print("\ncffi interface layer compiled successfully (NOTE: cython layer"
              " highly recommended for best performance)")
    else:
        print("\nERROR: cffi interface layer did NOT compile successfully")
        exit(1)

def use_cython():
    print("Will use cython interface layer")
    import subprocess
    global libcharm_interface
    libcharm_interface = 'cython'
    try:
        # remove charmlib_cython.c to force rebuild. cython and/or distutils don't rebuild .c
        # or obj files in some cases even when things have changed
        os.remove('charmpy/charmlib_cython.c')
    except:
        pass
    build_env = os.environ.copy()
    build_env['CFLAGS']  = '-I' + charm_include_path
    build_env['LDFLAGS'] = '-L' + libcharmPath
    cmd = [sys.executable, 'charmlib_cython_build.py', 'build_ext', '--build-lib', '__cython_objs__', '--build-temp', '__cython_objs__']
    p = subprocess.Popen(cmd, shell=False, cwd=os.getcwd() + '/charmpy', env=build_env)
    rc = p.wait()
    print('/////////////////////////////')
    if rc == 0:
        print("\ncython interface layer compiled successfully")
    else:
        print("\nERROR: cython interface layer did NOT compile successfully")
        exit(1)


if '--with-ctypes' in sys.argv:
    use_ctypes()
elif '--with-cffi' in sys.argv:
    assert(check_cffi())
    use_cffi()
elif '--with-cython' in sys.argv:
    assert(check_cython())
    use_cython()
else:
    if check_cython(): use_cython()
    elif check_cffi(): use_cffi()
    else: use_ctypes()

# -------------------------------------------------------------------------
# final step: generate charmpy config file

import json
config = {}
config['libcharm_interface'] = libcharm_interface
config['libcharm_path'] = libcharmPath
json.dump(config, open('charmpy/charmpy.cfg','w'))
print('charmpy.cfg file generated')

print('Setup complete\n')

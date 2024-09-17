import sys
import os
import re
import shutil
import platform
import subprocess
import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.install import install
from distutils.errors import DistutilsSetupError
from distutils.command.install_lib import install_lib as _install_lib
from distutils import log
import distutils

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

build_mpi = False



def get_build_machine():
    machine = platform.machine()
    if machine == 'arm64' or machine == 'aarch64':
        return 'arm8'
    return machine

def get_archflag_machine():
    machine = platform.machine()
    if machine == 'arm64' or machine == 'aarch64':
        return 'arm64'
    return machine

def get_build_os():
    os = platform.system()
    return os.lower()


def get_build_network_type(build_mpi):
    return 'netlrts' if not build_mpi else 'mpi'


def get_build_triple(build_mpi):
    return (get_build_machine(),
            get_build_os(),
            get_build_network_type(build_mpi)
            )


machine = get_build_machine()
system = get_build_os()


libcharm_filename2 = None
if system == 'windows' or system.startswith('cygwin'):
    libcharm_filename = 'charm.dll'
    libcharm_filename2 = 'charm.lib'
    charmrun_filename = 'charmrun.exe'
elif system == 'darwin':
    os.environ['ARCHFLAGS'] = f'-arch {get_archflag_machine()}'
    libcharm_filename = 'libcharm.dylib'
    charmrun_filename = 'charmrun'
    if 'CPPFLAGS' in os.environ:
        os.environ['CPPFLAGS'] += ' -Wno-error=implicit-function-declaration' # needed because some functions used by charm4py are not exported by charm.
    else:
        os.environ['CPPFLAGS'] = '-Wno-error=implicit-function-declaration '
else:  # Linux
    libcharm_filename = 'libcharm.so'
    charmrun_filename = 'charmrun'


try:
    charm4py_version = subprocess.check_output(['git', 'describe']).rstrip().decode().split('-')[0]
    if charm4py_version.startswith('v'):
        charm4py_version = charm4py_version[1:]
    with open(os.path.join('charm4py', '_version.py'), 'w') as f:
        f.write("version='" + charm4py_version + "'\n")
except:
    try:
        os.environ['PYTHONPATH'] = os.getcwd()
        os.environ['CHARM_NOLOAD'] = '1'
        from charm4py import _version
        charm4py_version = _version.version
    except:
        raise DistutilsSetupError('Could not determine Charm4py version')


def charm_built(charm_src_dir):
    library_path = os.path.join(charm_src_dir, 'charm', 'lib', libcharm_filename)
    if not os.path.exists(library_path):
        return False
    charmrun_path = os.path.join(charm_src_dir, 'charm', 'bin', charmrun_filename)
    if not os.path.exists(charmrun_path):
        return False
    return True


def check_libcharm_version(charm_src_dir):
    import ctypes
    library_path = os.path.join(charm_src_dir, 'charm', 'lib', libcharm_filename)
    lib = ctypes.CDLL(library_path)
    with open(os.path.join(os.getcwd(), 'charm4py', 'libcharm_version'), 'r') as f:
        req_version = tuple(int(n) for n in f.read().split('.'))
    commit_id_str = ctypes.c_char_p.in_dll(lib, "CmiCommitID").value.decode()
    version = [int(n) for n in commit_id_str.split('-')[0][1:].split('.')]
    try:
        version = tuple(version + [int(commit_id_str.split('-')[1])])
    except:
        version = tuple(version + [0])
    if version < req_version:
        req_str = '.'.join([str(n) for n in req_version])
        cur_str = '.'.join([str(n) for n in version])
        raise DistutilsSetupError('Charm++ version >= ' + req_str + ' required. '
                                  'Existing version is ' + cur_str)


def check_cffi():
    try:
        import cffi
        version_str = cffi.__version__.split('.')
        
        # pypy3.9 returns version string like '1.17.0.dev0'
        if (len(version_str) > 3): 
            version_str = version_str[:3]
            
        version = tuple(int(v) for v in version_str)
        if version < (1, 7):
            raise DistutilsSetupError('Charm4py requires cffi >= 1.7. '
                                      'Installed version is ' + cffi.__version__)
    except ImportError:
        raise DistutilsSetupError('cffi is not installed')


def build_libcharm(charm_src_dir, build_dir):

    lib_output_dirs = []
    charmrun_output_dirs = []
    lib_output_dirs.append(os.path.join(build_dir, 'charm4py', '.libs'))
    lib_output_dirs.append(os.path.join(os.getcwd(), 'charm4py', '.libs'))
    charmrun_output_dirs.append(os.path.join(build_dir, 'charmrun'))
    charmrun_output_dirs.append(os.path.join(os.getcwd(), 'charmrun'))
    for output_dir in (lib_output_dirs + charmrun_output_dirs):
        distutils.dir_util.mkpath(output_dir)

    if not os.path.exists(charm_src_dir) or not os.path.isdir(charm_src_dir):
        raise DistutilsSetupError('charm sources dir ' + charm_src_dir + ' not found')

    if not charm_built(charm_src_dir):

        if system == 'windows' or system.startswith('cygwin'):
            raise DistutilsSetupError('Building charm++ from setup.py not currently supported on Windows.'
                                      ' Please download a Charm4py binary wheel (64-bit Python required)')

        if os.path.exists(os.path.join(charm_src_dir, 'charm.tar.gz')):
            log.info('Uncompressing charm.tar.gz...')
            cmd = ['tar', 'xf', 'charm.tar.gz']
            p = subprocess.Popen(cmd, cwd=charm_src_dir, shell=False)
            rc = p.wait()
            if rc != 0:
                raise DistutilsSetupError('An error occured while building charm library')

        # divide by 2 to not hog the system. On systems with hyperthreading, this will likely
        # result in using same # cores as physical cores (therefore not all the logical cores)
        import multiprocessing
        build_num_cores = max(int(os.environ.get('CHARM_BUILD_PROCESSES', multiprocessing.cpu_count() // 2)), 1)
        extra_build_opts = os.environ.get('CHARM_EXTRA_BUILD_OPTS', '')

        target_machine, os_target, target_layer = get_build_triple(build_mpi)

        build_triple = f'{target_layer}-{os_target}-{target_machine}'
        cmd = f'./build charm4py {build_triple} -j{build_num_cores} --with-production {extra_build_opts}'
        print(cmd)

        p = subprocess.Popen(cmd.rstrip().split(' '),
                             cwd=os.path.join(charm_src_dir, 'charm'),
                             shell=False)
        rc = p.wait()
        if rc != 0:
            raise DistutilsSetupError('An error occured while building charm library')

        if system == 'darwin':
            old_file_path = os.path.join(charm_src_dir, 'charm', 'lib', 'libcharm.dylib')
            new_file_path = os.path.join(charm_src_dir, 'charm', 'lib', libcharm_filename)
            shutil.move(old_file_path, new_file_path)
            cmd = ['install_name_tool', '-id', '@rpath/../.libs/' + libcharm_filename, new_file_path]
            p = subprocess.Popen(cmd, shell=False)
            rc = p.wait()
            if rc != 0:
                raise DistutilsSetupError('install_name_tool error')

    # verify that the version of charm++ that was built is same or greater than the
    # one required by charm4py
    check_libcharm_version(charm_src_dir)

    # ---- copy libcharm ----
    lib_src_path = os.path.join(charm_src_dir, 'charm', 'lib', libcharm_filename)
    for output_dir in lib_output_dirs:
        log.info('copying ' + os.path.relpath(lib_src_path) + ' to ' + os.path.relpath(output_dir))
        shutil.copy(lib_src_path, output_dir)
    if libcharm_filename2 is not None:
        lib_src_path = os.path.join(charm_src_dir, 'charm', 'lib', libcharm_filename2)
        for output_dir in lib_output_dirs:
            log.info('copying ' + os.path.relpath(lib_src_path) + ' to ' + os.path.relpath(output_dir))
            shutil.copy(lib_src_path, output_dir)


    # ---- copy charmrun ----
    charmrun_src_path = os.path.join(charm_src_dir, 'charm', 'bin', charmrun_filename)
    for output_dir in charmrun_output_dirs:
        log.info('copying ' + os.path.relpath(charmrun_src_path) + ' to ' + os.path.relpath(output_dir))
        shutil.copy(charmrun_src_path, output_dir)


class custom_install(install, object):

    user_options = install.user_options + [
        ('mpi', None, 'Build libcharm with MPI')
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.mpi = False

    def finalize_options(self):
        global build_mpi
        if not build_mpi:
            build_mpi = bool(self.mpi)
        install.finalize_options(self)

    def run(self):
        install.run(self)


class custom_build_py(build_py, object):

    user_options = build_py.user_options + [
        ('mpi', None, 'Build libcharm with MPI')
    ]

    def initialize_options(self):
        build_py.initialize_options(self)
        self.mpi = False

    def finalize_options(self):
        global build_mpi
        if not build_mpi:
            build_mpi = bool(self.mpi)
        build_py.finalize_options(self)

    def run(self):
        if not self.dry_run:
            build_libcharm(os.path.join(os.getcwd(), 'charm_src'), self.build_lib)
            shutil.copy(os.path.join(os.getcwd(), 'LICENSE'), os.path.join(self.build_lib, 'charm4py'))
        super(custom_build_py, self).run()


class custom_build_ext(build_ext, object):

    user_options = build_ext.user_options + [
        ('mpi', None, 'Build libcharm with MPI')
    ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.mpi = False

    def finalize_options(self):
        global build_mpi
        if not build_mpi:
            build_mpi = bool(self.mpi)
        build_ext.finalize_options(self)

    def run(self):
        if not self.dry_run:
            build_libcharm(os.path.join(os.getcwd(), 'charm_src'), self.build_lib)
        super(custom_build_ext, self).run()

class _renameInstalled(_install_lib):
    def __init__(self, *args, **kwargs):
        _install_lib.__init__(self, *args, **kwargs)

    def install(self):
        log.info("Renaming libraries")
        outfiles = _install_lib.install(self)
        for file in outfiles:
            if "c_object_store" in file and system == "darwin":
                direc = os.path.dirname(file)
                install_name_command = "install_name_tool -change lib/libcharm.dylib "
                install_name_command += direc
                install_name_command += "/.libs/libcharm.dylib "
                install_name_command += direc
                install_name_command += "/c_object_store.*.so"
                log.info(install_name_command)
                os.system(install_name_command)
            elif "charmlib_cython" in file and system == "darwin":
                direc = os.path.dirname(file)
                install_name_command = "install_name_tool -change lib/libcharm.dylib "
                install_name_command += direc
                install_name_command += "/.libs/libcharm.dylib "
                install_name_command += direc
                install_name_command += "/charmlib_cython.*.so"
                log.info(install_name_command)
                os.system(install_name_command)
        return outfiles



extensions = []
py_impl = platform.python_implementation()

log.info("Check PyPy")
if py_impl == 'PyPy':
    os.environ['CHARM4PY_BUILD_CFFI'] = '1'

# elif 'CPY_WHEEL_BUILD_UNIVERSAL' not in os.environ:
else:
    log.info("Check sys version info")
    if sys.version_info[0] >= 3:
        log.info("Defining cython args")
        # compile C-extension module (from cython)
        from Cython.Build import cythonize
        my_include_dirs = []
        haveNumpy = False
        try:
            import numpy
            haveNumpy = True
            my_include_dirs.append(numpy.get_include())
        except:
            log.warn('WARNING: Building charmlib C-extension module without numpy support (numpy not found or import failed)')

        extra_link_args = []
        if os.name != 'nt':
            if system == 'darwin':
                extra_link_args=["-Wl,-rpath,@loader_path/../.libs"]
            else:
                extra_link_args=["-Wl,-rpath,$ORIGIN/../.libs"]

        cobject_extra_args = []
        log.info("Extra object args for object store")
        if os.name != 'nt':
            if system == 'darwin':
                cobject_extra_args=["-Wl,-rpath,@loader_path/.libs"]
            else:
                cobject_extra_args=["-Wl,-rpath,$ORIGIN/.libs"]

        extensions.extend(cythonize(setuptools.Extension('charm4py.charmlib.charmlib_cython',
                              sources=['charm4py/charmlib/charmlib_cython.pyx'],
                              include_dirs=['charm_src/charm/include'] + my_include_dirs,
                              library_dirs=[os.path.join(os.getcwd(), 'charm4py', '.libs')],
                              libraries=["charm"],
                              extra_compile_args=[],
                              extra_link_args=extra_link_args,
                              ), compile_time_env={'HAVE_NUMPY': haveNumpy}))

        extensions.extend(cythonize(setuptools.Extension('charm4py.c_object_store',
                              sources=['charm4py/c_object_store.pyx'],
                              include_dirs=['charm_src/charm/include'] + my_include_dirs,
                              library_dirs=[os.path.join(os.getcwd(), 'charm4py', '.libs')],
                              libraries=["charm"],
                              extra_compile_args=[],
                              extra_link_args=cobject_extra_args,
                              ), compile_time_env={'HAVE_NUMPY': haveNumpy}))
    else:
        try:
            check_cffi()
            os.environ['CHARM4PY_BUILD_CFFI'] = '1'
        except:
            pass


additional_setup_keywords = {}
if os.environ.get('CHARM4PY_BUILD_CFFI') == '1':
    check_cffi()
    additional_setup_keywords['cffi_modules'] = 'charm4py/charmlib/charmlib_cffi_build.py:ffibuilder'


setuptools.setup(
    version=charm4py_version,
    packages=setuptools.find_packages(),
    package_data={
        'charm4py': ['libcharm_version'],
    },
    ext_modules=extensions,
    cmdclass = {'build_py': custom_build_py,
                'build_ext': custom_build_ext,
                'install': custom_install,
                'install_lib': _renameInstalled,},
    **additional_setup_keywords
)

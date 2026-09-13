import sys
import os
import re
import shlex
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
enable_tracing = False


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
    configured_triple = os.environ.get('CHARM4PY_BUILD_TRIPLET')
    if configured_triple:
        return configured_triple
    return (get_build_machine(),
            get_build_os(),
            get_build_network_type(build_mpi)
            )


def get_charm_source_dir():
    configured_dir = os.environ.get('CHARM4PY_CHARM_DIR')
    if configured_dir:
        return os.path.abspath(os.path.expanduser(configured_dir))
    return os.path.join(os.getcwd(), 'charm_src', 'charm')


def get_charm_build_dir(charm_src_dir):
    configured_dir = os.environ.get('CHARM4PY_CHARM_BUILD_DIR')
    if configured_dir:
        return os.path.abspath(os.path.expanduser(configured_dir))
    return charm_src_dir


machine = get_build_machine()
system = get_build_os()
charm_src_dir = get_charm_source_dir()
charm_build_dir = get_charm_build_dir(charm_src_dir)


libcharm_filename2 = None
if system == 'windows' or system.startswith('cygwin'):
    libcharm_filename = 'charm.dll'
    libcharm_filename2 = 'charm.lib'
    charmrun_filename = 'charmrun.exe'
elif system == 'darwin':
    os.environ['ARCHFLAGS'] = f'-arch {get_archflag_machine()}'
    libcharm_filename = 'libcharm.dylib'
    packaged_libcharm_filename = 'libcharm4py.dylib'
    charmrun_filename = 'charmrun'
    if 'CPPFLAGS' in os.environ:
        os.environ['CPPFLAGS'] += ' -Wno-error=implicit-function-declaration' # needed because some functions used by charm4py are not exported by charm.
    else:
        os.environ['CPPFLAGS'] = '-Wno-error=implicit-function-declaration '
else:  # Linux
    libcharm_filename = 'libcharm.so'
    charmrun_filename = 'charmrun'

if system != 'darwin':
    packaged_libcharm_filename = libcharm_filename


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


def charm_built(charm_build_dir):
    library_path = os.path.join(charm_build_dir, 'lib', libcharm_filename)
    if not os.path.exists(library_path):
        return False
    charmrun_path = os.path.join(charm_build_dir, 'bin', charmrun_filename)
    if not os.path.exists(charmrun_path):
        return False
    return True


def check_libcharm_version(charm_build_dir):
    import ctypes
    library_path = os.path.join(charm_build_dir, 'lib', libcharm_filename)
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


def prepare_darwin_libraries(charm_build_dir):
    """Make a Charm++/Reconverse build relocatable with Charm4py."""
    library_dir = os.path.join(charm_build_dir, 'lib')
    libcharm_path = os.path.join(library_dir, libcharm_filename)
    runtime_name_map = {
        'libreconverse.dylib': 'libcharm4py_reconverse.dylib',
        'liblci.dylib': 'libcharm4py_lci.dylib',
        'liblct.dylib': 'libcharm4py_lct.dylib',
    }
    runtime_paths = [os.path.join(library_dir, name)
                     for name in runtime_name_map]
    existing_runtime_paths = [path for path in runtime_paths
                              if os.path.isfile(path)]

    if existing_runtime_paths and len(existing_runtime_paths) != len(runtime_paths):
        missing = [path for path in runtime_paths if not os.path.isfile(path)]
        raise DistutilsSetupError(
            'Incomplete Reconverse runtime in ' + library_dir + ': missing ' +
            ', '.join(os.path.basename(path) for path in missing))

    bundled_runtime_paths = []
    for source_path in existing_runtime_paths:
        bundled_name = runtime_name_map[os.path.basename(source_path)]
        bundled_path = os.path.join(library_dir, bundled_name)
        shutil.copy2(source_path, bundled_path)
        bundled_runtime_paths.append(bundled_path)

    libraries = [libcharm_path] + bundled_runtime_paths
    for library_path in libraries:
        if library_path == libcharm_path:
            install_id = '@rpath/../.libs/' + packaged_libcharm_filename
        else:
            install_id = '@loader_path/' + os.path.basename(library_path)
        subprocess.check_call(['install_name_tool', '-id', install_id, library_path])

    # Reconverse and LCI normally use @rpath install names. Binding these
    # dependencies to the containing directory avoids accidentally loading a
    # different Reconverse installation from the host process's search paths.
    for library_path in libraries:
        dependencies = subprocess.check_output(
            ['otool', '-L', library_path], text=True).splitlines()[1:]
        dependency_paths = [line.strip().split(' (compatibility')[0]
                            for line in dependencies]
        for dependency_path in dependency_paths:
            dependency_name = os.path.basename(dependency_path)
            original_name = next(
                (name for name, bundled_name in runtime_name_map.items()
                 if dependency_name == name or dependency_name == bundled_name),
                None)
            if original_name is not None:
                replacement = ('@loader_path/' +
                               runtime_name_map[original_name])
                if dependency_path != replacement:
                    subprocess.check_call([
                        'install_name_tool', '-change', dependency_path,
                        replacement, library_path])

    return bundled_runtime_paths


def build_libcharm(charm_src_dir, charm_build_dir, build_dir):

    configured_triple = get_build_triple(build_mpi)
    if isinstance(configured_triple, str):
        build_triple = configured_triple
    else:
        target_machine, os_target, target_layer = configured_triple
        build_triple = f'{target_layer}-{os_target}-{target_machine}'
    is_reconverse_build = build_triple.startswith('reconverse-')

    lib_output_dirs = []
    charmrun_output_dirs = []
    lib_output_dirs.append(os.path.join(build_dir, 'charm4py', '.libs'))
    lib_output_dirs.append(os.path.join(os.getcwd(), 'charm4py', '.libs'))
    charmrun_output_dirs.append(os.path.join(build_dir, 'charmrun'))
    charmrun_output_dirs.append(os.path.join(os.getcwd(), 'charmrun'))
    for output_dir in (lib_output_dirs + charmrun_output_dirs):
        distutils.dir_util.mkpath(output_dir)

    # Source distributions carry a compressed Charm++ tree instead of an
    # expanded charm_src/charm directory.
    charm_archive = os.path.join(os.path.dirname(charm_src_dir), 'charm.tar.gz')
    if not os.path.isdir(charm_src_dir) and os.path.isfile(charm_archive):
        log.info('Uncompressing charm.tar.gz...')
        cmd = ['tar', 'xf', os.path.basename(charm_archive)]
        p = subprocess.Popen(cmd, cwd=os.path.dirname(charm_archive), shell=False)
        rc = p.wait()
        if rc != 0:
            raise DistutilsSetupError('An error occured while building charm library')

    if not os.path.exists(charm_src_dir) or not os.path.isdir(charm_src_dir):
        raise DistutilsSetupError('charm sources dir ' + charm_src_dir + ' not found')

    if not charm_built(charm_build_dir):

        if system == 'windows' or system.startswith('cygwin'):
            raise DistutilsSetupError('Building charm++ from setup.py not currently supported on Windows.'
                                      ' Please download a Charm4py binary wheel (64-bit Python required)')

        # divide by 2 to not hog the system. On systems with hyperthreading, this will likely
        # result in using same # cores as physical cores (therefore not all the logical cores)
        import multiprocessing
        build_num_cores = max(int(os.environ.get('CHARM_BUILD_PROCESSES', multiprocessing.cpu_count() // 2)), 1)
        extra_build_opts = os.environ.get('CHARM_EXTRA_BUILD_OPTS', '')

        if enable_tracing:
         extra_build_opts += " --enable-tracing "
        
        extra_build_args = shlex.split(extra_build_opts)
        cmd = ['./build', 'charm4py', build_triple,
               f'-j{build_num_cores}', '--with-production']
        if charm_build_dir != charm_src_dir:
            cmd.append('--destination=' + charm_build_dir)

        if is_reconverse_build:
            if '--disable-fortran' not in extra_build_args:
                cmd.append('--disable-fortran')

            local_reconverse_dir = os.path.join(charm_src_dir, 'reconverse')
            if (os.path.isdir(local_reconverse_dir) and
                    not any(arg.startswith('--with-fetch-reconverse-')
                            for arg in extra_build_args)):
                cmd.append('--with-fetch-reconverse-dir=' +
                           local_reconverse_dir)

            configured_lci_dir = os.environ.get('CHARM4PY_LCI_DIR')
            if configured_lci_dir:
                local_lci_dir = os.path.abspath(
                    os.path.expanduser(configured_lci_dir))
            else:
                local_lci_dir = os.path.join(
                    charm_src_dir, build_triple, '_deps', 'lci-src')
            if os.path.isdir(local_lci_dir):
                cmd.append('--with-cmake-args=' +
                           '-DFETCHCONTENT_SOURCE_DIR_LCI=' + local_lci_dir)

        cmd.extend(extra_build_args)
        log.info('building Charm++: ' + shlex.join(cmd))

        p = subprocess.Popen(cmd,
                             cwd=charm_src_dir,
                             shell=False)
        rc = p.wait()
        if rc != 0:
            raise DistutilsSetupError('An error occured while building charm library')

    runtime_lib_src_paths = []
    if system == 'darwin':
        # Normalize prebuilt libraries too. Extension modules record the
        # libcharm ID at link time, and Reconverse's dependencies must remain
        # colocated after Charm4py is installed.
        try:
            runtime_lib_src_paths = prepare_darwin_libraries(charm_build_dir)
        except subprocess.CalledProcessError as error:
            raise DistutilsSetupError('install_name_tool error') from error

    # verify that the version of charm++ that was built is same or greater than the
    # one required by charm4py
    check_libcharm_version(charm_build_dir)

    # ---- copy libcharm and its colocated runtime libraries ----
    lib_src_path = os.path.join(charm_build_dir, 'lib', libcharm_filename)
    for source_path in [lib_src_path] + runtime_lib_src_paths:
        for output_dir in lib_output_dirs:
            log.info('copying ' + os.path.relpath(source_path) + ' to ' + os.path.relpath(output_dir))
            shutil.copy(source_path, output_dir)
    bundled_runtime_names = [
        'libcharm4py_reconverse.dylib', 'libcharm4py_lci.dylib',
        'libcharm4py_lct.dylib']
    if not runtime_lib_src_paths:
        for output_dir in lib_output_dirs:
            for filename in bundled_runtime_names:
                stale_path = os.path.join(output_dir, filename)
                if os.path.isfile(stale_path):
                    os.unlink(stale_path)
    if packaged_libcharm_filename != libcharm_filename:
        for output_dir in lib_output_dirs:
            packaged_path = os.path.join(output_dir,
                                         packaged_libcharm_filename)
            log.info('copying ' + os.path.relpath(lib_src_path) + ' to ' +
                     os.path.relpath(packaged_path))
            shutil.copy(lib_src_path, packaged_path)
    for output_dir in lib_output_dirs:
        marker_path = os.path.join(output_dir, 'reconverse')
        if is_reconverse_build:
            with open(marker_path, 'w'):
                pass
        elif os.path.isfile(marker_path):
            os.unlink(marker_path)
    if libcharm_filename2 is not None:
        lib_src_path = os.path.join(charm_build_dir, 'lib', libcharm_filename2)
        for output_dir in lib_output_dirs:
            log.info('copying ' + os.path.relpath(lib_src_path) + ' to ' + os.path.relpath(output_dir))
            shutil.copy(lib_src_path, output_dir)


    # ---- copy charmrun ----
    charmrun_src_path = os.path.join(charm_build_dir, 'bin', charmrun_filename)
    for output_dir in charmrun_output_dirs:
        log.info('copying ' + os.path.relpath(charmrun_src_path) + ' to ' + os.path.relpath(output_dir))
        shutil.copy(charmrun_src_path, output_dir)


class custom_install(install, object):

    user_options = install.user_options + [
        ('mpi', None, 'Build libcharm with MPI'),
        ('enable-tracing', None, 'Build libcharm with tracing enabled')
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.mpi = False
        self.enable_tracing = False

    def finalize_options(self):
        global build_mpi
        if not build_mpi:
            build_mpi = bool(self.mpi)

        global enable_tracing
        if not enable_tracing:
            enable_tracing = bool(self.enable_tracing)
        install.finalize_options(self)

    def run(self):
        install.run(self)


class custom_build_py(build_py, object):

    user_options = build_py.user_options + [
        ('mpi', None, 'Build libcharm with MPI'),
        ('enable-tracing', None, 'Build libcharm with tracing enabled')
    ]

    def initialize_options(self):
        build_py.initialize_options(self)
        self.mpi = False
        self.enable_tracing = False

    def finalize_options(self):
        global build_mpi
        if not build_mpi:
            build_mpi = bool(self.mpi)
        global enable_tracing
        if not enable_tracing:
            enable_tracing = bool(self.enable_tracing)
        build_py.finalize_options(self)

    def run(self):
        if not self.dry_run:
            build_libcharm(charm_src_dir, charm_build_dir, self.build_lib)
            shutil.copy(os.path.join(os.getcwd(), 'LICENSE'), os.path.join(self.build_lib, 'charm4py'))
        super(custom_build_py, self).run()


class custom_build_ext(build_ext, object):

    user_options = build_ext.user_options + [
        ('mpi', None, 'Build libcharm with MPI'),
        ('enable-tracing', None, 'Build libcharm with tracing enabled')
    ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.mpi = False
        self.enable_tracing = False

    def finalize_options(self):
        global build_mpi
        if not build_mpi:
            build_mpi = bool(self.mpi)

        global enable_tracing
        if not enable_tracing:
            enable_tracing = bool(self.enable_tracing)
        build_ext.finalize_options(self)

    def run(self):
        if not self.dry_run:
            build_libcharm(charm_src_dir, charm_build_dir, self.build_lib)
        super(custom_build_ext, self).run()

class _renameInstalled(_install_lib):
    def __init__(self, *args, **kwargs):
        _install_lib.__init__(self, *args, **kwargs)

    
    def install(self):
        log.info("Renaming libraries")
        outfiles = _install_lib.install(self)
        '''
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
                install_name_command += "/../.libs/libcharm.dylib "
                install_name_command += direc
                install_name_command += "/charmlib_cython.*.so"
                log.info(install_name_command)
                os.system(install_name_command)
        '''
        return outfiles



extensions = []
py_impl = platform.python_implementation()



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
            
    cudaBuild = os.environ.get('CHARM_EXTRA_BUILD_OPTS', '').find('cuda') != -1
    
    extensions.extend(cythonize(setuptools.Extension('charm4py.charmlib.charmlib_cython',
                            sources=['charm4py/charmlib/charmlib_cython.pyx'],
                            include_dirs=[os.path.join(charm_build_dir, 'include')] + my_include_dirs,
                            library_dirs=[os.path.join(os.getcwd(), 'charm4py', '.libs')],
                            libraries=["charm"],
                            extra_compile_args=[],
                            extra_link_args=extra_link_args,
                            ), compile_time_env={'HAVE_NUMPY': haveNumpy,
                                                 'HAVE_CUDA_BUILD': cudaBuild}))

    extensions.extend(cythonize(setuptools.Extension('charm4py.c_object_store',
                            sources=['charm4py/c_object_store.pyx'],
                            include_dirs=[os.path.join(charm_build_dir, 'include')] + my_include_dirs,
                            library_dirs=[os.path.join(os.getcwd(), 'charm4py', '.libs')],
                            libraries=["charm"],
                            extra_compile_args=[],
                            extra_link_args=cobject_extra_args,
                            ), compile_time_env={'HAVE_NUMPY': haveNumpy,
                                                 'HAVE_CUDA_BUILD': cudaBuild}))


additional_setup_keywords = {}
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

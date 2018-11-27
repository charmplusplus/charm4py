import sys
import os
import shutil
import platform
import subprocess
import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from distutils.errors import DistutilsSetupError
from distutils import log
import distutils


system = platform.system()
libcharm_filename2 = None
if system == 'Windows' or system.lower().startswith('cygwin'):
    libcharm_filename = 'charm.dll'
    libcharm_filename2 = 'charm.lib'
    charmrun_filename = 'charmrun.exe'
elif system == 'Darwin':
    os.environ['ARCHFLAGS'] = '-arch x86_64'
    libcharm_filename = 'libcharm.dylib'
    charmrun_filename = 'charmrun'
else:
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
        raise DistutilsSetupError('Could not determine charm4py version')


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
        version = tuple(int(v) for v in cffi.__version__.split('.'))
        if version < (1, 7):
            raise DistutilsSetupError('charm4py requires cffi >= 1.7. '
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

        if system == 'Windows' or system.lower().startswith('cygwin'):
            raise DistutilsSetupError('Building charm++ from setup.py not currently supported on Windows.'
                                      ' Please download a charm4py binary wheel (64-bit Python required)')

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
        build_num_cores = int(os.environ.get('CHARM_BUILD_PROCESSES', multiprocessing.cpu_count() // 2))
        extra_build_opts = os.environ.get('CHARM_EXTRA_BUILD_OPTS', '')
        if system == 'Darwin':
            cmd = './build charm4py netlrts-darwin-x86_64 tcp -j' + str(build_num_cores) + ' --with-production ' + extra_build_opts
        else:
            cmd = './build charm4py netlrts-linux-x86_64 tcp -j' + str(build_num_cores) + ' --with-production ' + extra_build_opts
        p = subprocess.Popen(cmd.rstrip().split(' '),
                             cwd=os.path.join(charm_src_dir, 'charm'),
                             shell=False)
        rc = p.wait()
        if rc != 0:
            raise DistutilsSetupError('An error occured while building charm library')

        if system == 'Darwin':
            old_file_path = os.path.join(charm_src_dir, 'charm', 'lib', 'libcharm.so')
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


class specialized_build_py(build_py, object):

    def run(self):
        if not self.dry_run:
            build_libcharm(os.path.join(os.getcwd(), 'charm_src'), self.build_lib)
            shutil.copy(os.path.join(os.getcwd(), 'LICENSE'), os.path.join(self.build_lib, 'charm4py'))
        super(specialized_build_py, self).run()


class specialized_build_ext(build_ext, object):

    def run(self):
        if not self.dry_run:
            build_libcharm(os.path.join(os.getcwd(), 'charm_src'), self.build_lib)
        super(specialized_build_ext, self).run()


extensions = []
py_impl = platform.python_implementation()

if py_impl == 'PyPy':
    os.environ['CHARM4PY_BUILD_CFFI'] = '1'
elif 'CPY_WHEEL_BUILD_UNIVERSAL' not in os.environ:
    if sys.version_info[0] >= 3:
        # compile C-extension module (from cython)
        from Cython.Build import cythonize
        my_include_dirs = []
        haveNumpy = False
        try:
            import numpy
            haveNumpy = True
            my_include_dirs.append(numpy.get_include())
        except:
            log.warn('WARNING: Building charmlib C-extension module without numpy support (numpy not found)')

        extra_link_args = []
        if os.name != 'nt':
            if system == 'Darwin':
                extra_link_args=["-Wl,-rpath,@loader_path/../.libs"]
            else:
                extra_link_args=["-Wl,-rpath,$ORIGIN/../.libs"]

        extensions.extend(cythonize(setuptools.Extension('charm4py.charmlib.charmlib_cython',
                              sources=['charm4py/charmlib/charmlib_cython.pyx'],
                              include_dirs=['charm_src/charm/include'] + my_include_dirs,
                              library_dirs=[os.path.join(os.getcwd(), 'charm4py', '.libs')],
                              libraries=["charm"],
                              extra_compile_args=['-g0', '-O3'],
                              extra_link_args=extra_link_args,
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


with open('README.rst', 'r') as f:
    long_description = f.read()


setuptools.setup(
    name='charm4py',
    version=charm4py_version,
    author='Juan Galvez and individual contributors',
    author_email='jjgalvez@illinois.edu',
    description='charm4py Parallel Programming Framework',
    long_description=long_description,
    url='https://github.com/UIUC-PPL/charm4py',
    keywords='parallel parallel-programming distributed distributed-computing hpc HPC runtime',
    packages=setuptools.find_packages(),
    package_data={
        'charm4py': ['libcharm_version'],
    },
    entry_points={
        'console_scripts': [
            'charmrun = charmrun.start:start',
        ],
    },
    install_requires=['numpy>=1.10.0'],
    #python_requires='>=2.7, ~=3.4',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: Free for non-commercial use',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: System :: Distributed Computing',
        'Topic :: System :: Clustering',
    ],
    ext_modules=extensions,
    cmdclass = {'build_py': specialized_build_py,
                'build_ext': specialized_build_ext},
    **additional_setup_keywords
)

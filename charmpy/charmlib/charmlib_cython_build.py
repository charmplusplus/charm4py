from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

my_include_dirs = []
HAVE_NUMPY = False
try:
    import numpy
    HAVE_NUMPY = True
    my_include_dirs.append(numpy.get_include())
except:
    print("Building cython libcharm wrapper without numpy support (numpy not found)")


setup(
    ext_modules = cythonize([
    Extension("charmlib_cython", ["charmlib_cython.pyx"], libraries=["charm"],
              include_dirs=my_include_dirs, extra_compile_args=["-g0", "-O3"])
    ], compile_time_env={'HAVE_NUMPY': HAVE_NUMPY})
)

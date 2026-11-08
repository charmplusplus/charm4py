# Build of the charm4py._ckdyn extension module (SPEC-4a.md section 4).
# setup.py and charm4py/charmlib are not used by this build.
cmake_minimum_required(VERSION 3.18...3.29)
project(charm4py_ckdyn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- libcharm --------------------------------------------------------------
set(CHARM4PY_CHARM_BUILD_DIR "$ENV{CHARM4PY_CHARM_BUILD_DIR}" CACHE PATH
    "Charm++ build directory (lib/libcharm.*, include/ckdyn.h)")
if(NOT CHARM4PY_CHARM_BUILD_DIR)
  if(DEFINED ENV{CHARM4PY_CHARM_DIR})
    message(FATAL_ERROR
      "CHARM4PY_CHARM_BUILD_DIR is not set. Building Charm++ from here is not supported yet: "
      "in $ENV{CHARM4PY_CHARM_DIR} run\n"
      "  ./build charm4py <triplet> -j8 --with-production --disable-fortran --with-refnum-type=\"unsigned int\"\n"
      "and set CHARM4PY_CHARM_BUILD_DIR to the resulting build directory.")
  else()
    message(FATAL_ERROR "Set CHARM4PY_CHARM_BUILD_DIR to a Charm++ charm4py build directory.")
  endif()
endif()
get_filename_component(CHARM_BUILD "${CHARM4PY_CHARM_BUILD_DIR}" ABSOLUTE)
if(NOT EXISTS "${CHARM_BUILD}/include/ckdyn.h")
  message(FATAL_ERROR "${CHARM_BUILD}/include/ckdyn.h not found")
endif()
if(APPLE)
  set(LIBCHARM_SRC "${CHARM_BUILD}/lib/libcharm.dylib")
  set(LIBCHARM_STAGED_NAME "libcharm4py.dylib")
else()
  set(LIBCHARM_SRC "${CHARM_BUILD}/lib/libcharm.so")
  set(LIBCHARM_STAGED_NAME "libcharm.so")
endif()
if(NOT EXISTS "${LIBCHARM_SRC}")
  message(FATAL_ERROR "${LIBCHARM_SRC} not found")
endif()

# ---- Python, nanobind, numpy, greenlet ---------------------------------------
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NUMPY_INCLUDE RESULT_VARIABLE rc)
if(rc)
  message(FATAL_ERROR "numpy is required to build charm4py")
endif()
execute_process(COMMAND "${Python_EXECUTABLE}" -c
                "import greenlet, os; print(os.path.dirname(os.path.dirname(greenlet.__file__)))"
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE GREENLET_INCLUDE RESULT_VARIABLE rc)
if(rc)
  message(FATAL_ERROR "greenlet is required to build charm4py (its C API header greenlet/greenlet.h)")
endif()

# ---- bundle libcharm and the runtime libraries into .libs --------------------
set(LIBS_STAGE "${CMAKE_CURRENT_BINARY_DIR}/libs")
set(BUNDLE_CMD "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/bundle_libs.py"
    --charm-build-dir "${CHARM_BUILD}" --out "${LIBS_STAGE}"
    --also "${CMAKE_CURRENT_SOURCE_DIR}/charm4py/.libs")
# at configure time, so the staged libcharm exists to link against
execute_process(COMMAND ${BUNDLE_CMD} RESULT_VARIABLE rc)
if(rc)
  message(FATAL_ERROR "tools/bundle_libs.py failed")
endif()
# and again whenever libcharm changes
add_custom_command(OUTPUT "${LIBS_STAGE}/.stamp"
                   COMMAND ${BUNDLE_CMD}
                   COMMAND ${CMAKE_COMMAND} -E touch "${LIBS_STAGE}/.stamp"
                   DEPENDS "${LIBCHARM_SRC}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/bundle_libs.py"
                   COMMENT "Bundling libcharm into .libs")
add_custom_target(bundle_libs DEPENDS "${LIBS_STAGE}/.stamp")

# ---- the module --------------------------------------------------------------
set(CKDYN_SOURCES src/ckdyn_py.cpp)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/objstore_py.cpp")
  list(APPEND CKDYN_SOURCES src/objstore_py.cpp)
else()
  message(FATAL_ERROR "src/objstore_py.cpp is missing")
endif()
nanobind_add_module(_ckdyn NB_STATIC ${CKDYN_SOURCES})
add_dependencies(_ckdyn bundle_libs)
target_include_directories(_ckdyn PRIVATE "${CHARM_BUILD}/include" "${NUMPY_INCLUDE}" "${GREENLET_INCLUDE}")
target_link_libraries(_ckdyn PRIVATE "${LIBS_STAGE}/${LIBCHARM_STAGED_NAME}")
if(APPLE)
  set_target_properties(_ckdyn PROPERTIES BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "@loader_path/.libs")
else()
  set_target_properties(_ckdyn PROPERTIES BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "$ORIGIN/.libs")
endif()

install(TARGETS _ckdyn LIBRARY DESTINATION charm4py)
install(DIRECTORY "${LIBS_STAGE}/" DESTINATION charm4py/.libs PATTERN ".stamp" EXCLUDE PATTERN ".tmp" EXCLUDE)

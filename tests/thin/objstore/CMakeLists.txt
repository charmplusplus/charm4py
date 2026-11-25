# Standalone compile test for src/objstore_py.cpp (not part of the package build).
cmake_minimum_required(VERSION 3.18)
project(objstore_test CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(_objstore_test
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/objstore_py.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/module.cpp)

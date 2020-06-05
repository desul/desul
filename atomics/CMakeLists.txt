# CMakeList.txt : CMake project for DOEAtomicSupport, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)
find_package(OpenMP)

MACRO(APPEND_GLOB VAR)
  FILE(GLOB LOCAL_TMP_VAR ${ARGN})
  LIST(APPEND ${VAR} ${LOCAL_TMP_VAR})
ENDMACRO()

# APPEND_GLOB(DESUL_ATOMIC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
blt_add_library(NAME desul_atomics
                SOURCES src/Lock_Array_CUDA.cpp
                INCLUDES include
                DEPENDS_ON cuda)
# TODO: Add tests and install targets if needed.
add_subdirectory(unit_tests)
add_subdirectory(perf_tests)

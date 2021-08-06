cmake_minimum_required (VERSION 3.10)

project(DESUL CXX)

# Default to C++14 if not set TODO: check to ensure this is the minimum
if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++14" CACHE STRING "")
endif()

include(cmake/blt_boilerplate.cmake)

cmake_dependent_option(DESUL_ENABLE_TESTS "Build tests" On "ENABLE_TESTS" Off)

# list of backends to add as dependencies
if(ENABLE_OPENMP)
  list(APPEND DESUL_BACKENDS openmp)
endif()
if(ENABLE_CUDA OR ENABLE_CLANG_CUDA)
  list(APPEND DESUL_BACKENDS cuda)
endif()
if(ENABLE_HIP)
  list(APPEND DESUL_BACKENDS hip)
endif()

MACRO(APPEND_GLOB VAR)
  FILE(GLOB LOCAL_TMP_VAR ${ARGN})
  LIST(APPEND ${VAR} ${LOCAL_TMP_VAR})
ENDMACRO()

add_subdirectory(atomics)

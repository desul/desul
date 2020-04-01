# CMakeList.txt : CMake project for DOEAtomicSupport, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)
find_package(OpenMP)

MACRO(APPEND_GLOB VAR)
  FILE(GLOB LOCAL_TMP_VAR ${ARGN})
  LIST(APPEND ${VAR} ${LOCAL_TMP_VAR})
ENDMACRO()

include_directories("include")
# Add source to this project's executable.
SET(DESUL_ATOMIC_SOURCES 
  src/Lock_Array.cpp
  src/Lock_Array_CUDA.cpp
)
# APPEND_GLOB(DESUL_ATOMIC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(desul_atomics ${DESUL_ATOMIC_SOURCES})
# TODO: Add tests and install targets if needed.

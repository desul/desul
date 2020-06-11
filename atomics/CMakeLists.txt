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
if(ENABLE_CUDA)
  set(DESUL_ATOMICS_SOURCES src/Lock_Array_CUDA.cpp)
  set(SOURCES_ARG SOURCES ${DESUL_ATOMICS_SOURCES})
  set(DEPENDS_ARG DEPENDS_ON cuda)
endif()
blt_add_library(NAME desul_atomics
                HEADERS include/desul/atomics.hpp
                ${SOURCES_ARG}
                INCLUDES include
                ${DEPENDS_ARG})

# TODO: Add tests and install targets if needed.
add_subdirectory(unit_tests)
add_subdirectory(performance_tests)

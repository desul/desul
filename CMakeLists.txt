cmake_minimum_required (VERSION 3.10)

# Default to C++14 if not set so GTest/GMock can build
if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++14" CACHE STRING "")
endif()

include(external/blt/SetupBLT.cmake)

MACRO(APPEND_GLOB VAR)
  FILE(GLOB LOCAL_TMP_VAR ${ARGN})
  LIST(APPEND ${VAR} ${LOCAL_TMP_VAR})
ENDMACRO()

add_subdirectory(atomics)

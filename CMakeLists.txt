cmake_minimum_required (VERSION 3.10)
find_package(OpenMP)

MACRO(APPEND_GLOB VAR)
  FILE(GLOB LOCAL_TMP_VAR ${ARGN})
  LIST(APPEND ${VAR} ${LOCAL_TMP_VAR})
ENDMACRO()

add_subdirectory(atomics)

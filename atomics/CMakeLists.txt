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
  set(DESUL_CUDA_ATOMICS_SOURCES src/Lock_Array_CUDA.cpp)
  set(SOURCES_ARG SOURCES ${DESUL_CUDA_ATOMICS_SOURCES})
  set(DEPENDS_ARG DEPENDS_ON cuda)
endif()
if(ENABLE_HIP)
  set(DESUL_HIP_ATOMICS_SOURCES src/Lock_Array_HIP.cpp)
  set(SOURCES_ARG SOURCES ${DESUL_HIP_ATOMICS_SOURCES})
  set(DEPENDS_ARG DEPENDS_ON hip)
endif()

set(INCLUDES_ARG
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

blt_add_library(NAME desul_atomics
                HEADERS include/desul/atomics.hpp
                ${SOURCES_ARG}
                INCLUDES ${INCLUDES_ARG}
                ${DEPENDS_ARG})

# Install options
install(TARGETS desul_atomics
  EXPORT desul_atomics
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)

install(EXPORT desul_atomics
  DESTINATION lib/cmake/desul)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/atomics/include/
  DESTINATION include
  FILES_MATCHING PATTERN *.hpp)


  # TODO: Add tests and install targets if needed.
add_subdirectory(unit_tests)
add_subdirectory(performance_tests)

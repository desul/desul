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
  set(DESUL_ATOMICS_ENABLE_CUDA ON)
  set(DESUL_CUDA_ATOMICS_SOURCES src/Lock_Array_CUDA.cpp)
  set(SOURCES_ARG ${DESUL_CUDA_ATOMICS_SOURCES})
  # FIXME_BLT
  if(TARGET cuda AND NOT TARGET blt::cuda)
    add_library(blt::cuda ALIAS cuda)
  endif()
  set(DEPENDS_ARG DEPENDS_ON blt::cuda)
endif()
if(CUDA_SEPARABLE_COMPILATION) # for compatibility with BLT
  set(DESUL_ATOMICS_ENABLE_CUDA_SEPARABLE_COMPILATION ON)
endif()
if(ENABLE_HIP)
  set(DESUL_ATOMICS_ENABLE_HIP ON)
  set(DESUL_HIP_ATOMICS_SOURCES src/Lock_Array_HIP.cpp)
  set(SOURCES_ARG ${DESUL_HIP_ATOMICS_SOURCES})
  set(DEPENDS_ARG DEPENDS_ON blt::hip)
endif()
cmake_dependent_option(HIP_SEPARABLE_COMPILATION "Enable separate compilation of device code" OFF "ENABLE_HIP" OFF) # does not exist in BLT at the moment
if(HIP_SEPARABLE_COMPILATION)
  set(DESUL_ATOMICS_ENABLE_HIP_SEPARABLE_COMPILATION ON)
endif()
option(ENABLE_SYCL "Enable SYCL support" OFF) # option does not actually yet exist on the BLT side
if(ENABLE_SYCL)
  set(DESUL_ATOMICS_ENABLE_SYCL ON)
  set(DESUL_SYCL_ATOMICS_SOURCES src/Lock_Array_SYCL.cpp)
  set(SOURCES_ARG ${DESUL_SYCL_ATOMICS_SOURCES})
endif()
cmake_dependent_option(SYCL_SEPARABLE_COMPILATION "Enable separate compilation of device code" OFF "ENABLE_SYCL" OFF) # does not exist in BLT at the moment
if(SYCL_SEPARABLE_COMPILATION)
  set(DESUL_ATOMICS_ENABLE_SYCL_SEPARABLE_COMPILATION ON)
endif()
if(ENABLE_OPENMP)
  set(DESUL_ATOMICS_ENABLE_OPENMP ON)
endif()
configure_file(
  ${PROJECT_SOURCE_DIR}/atomics/Config.hpp.cmake.in
  ${PROJECT_BINARY_DIR}/include/desul/atomics/Config.hpp)

set(INCLUDES_ARG
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

blt_add_library(NAME desul_atomics
                HEADERS include/desul/atomics.hpp
                SOURCES src/common.cpp ${SOURCES_ARG}
                INCLUDES ${INCLUDES_ARG}
                ${DEPENDS_ARG})
target_include_directories (desul_atomics PUBLIC
  $<BUILD_INTERFACE:${PROJECT_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/atomics/desul_atomicsConfig.cmake.in
  ${PROJECT_BINARY_DIR}/desul_atomicsConfig.cmake
  PATH_VARS CMAKE_INSTALL_PREFIX
  INSTALL_DESTINATION lib/cmake/desul)

install(FILES
  ${PROJECT_BINARY_DIR}/desul_atomicsConfig.cmake
  DESTINATION lib/cmake/desul)

write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/desul_atomics-config-version.cmake
  COMPATIBILITY SameMajorVersion)

install(FILES
  "${PROJECT_BINARY_DIR}/desul_atomics-config-version.cmake"
  DESTINATION lib/cmake/desul)

# Install options
install(TARGETS desul_atomics
  EXPORT desul_atomics
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)

install(EXPORT desul_atomics
  DESTINATION lib/cmake/desul)

install(DIRECTORY
  ${PROJECT_SOURCE_DIR}/atomics/include/
  ${PROJECT_BINARY_DIR}/include/
  DESTINATION include
  FILES_MATCHING
  PATTERN *.hpp
  PATTERN *.inc
  PATTERN *.inc_*)

add_subdirectory(unit_tests)
add_subdirectory(performance_tests)

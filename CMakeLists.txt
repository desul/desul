cmake_minimum_required (VERSION 3.10)

project(desul
  LANGUAGES CXX
  VERSION 0.1.0)

# Default to C++14 if not set TODO: check to ensure this is the minimum
if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++14" CACHE STRING "")
endif()

set(DESUL_DEPENDENT_OPTIONS
  ENABLE_OPENMP
  ENABLE_CUDA
  ENABLE_HIP
  ENABLE_SYCL
  ENABLE_TESTS
)
foreach (option ${DESUL_DEPENDENT_OPTIONS})
   if (${DESUL_${option}})
      if (NOT ${option})
         message(FATAL_ERROR "DESUL_${option} set to On, but ${option} is Off. Please set ${option} to On enable this feature.")
      endif ()
   endif ()
endforeach ()


include(CMakeDependentOption)
option(ENABLE_SYCL "Enable SYCL support" OFF) # option does not actually yet exist on the BLT side
cmake_dependent_option(DESUL_ENABLE_CUDA "Build with CUDA support" On "ENABLE_CUDA" Off)
cmake_dependent_option(DESUL_ENABLE_HIP "Build with HIP support" On "ENABLE_HIP" Off)
cmake_dependent_option(DESUL_ENABLE_OPENMP "Build with OpenMP support" On "ENABLE_OPENMP" Off)
cmake_dependent_option(DESUL_ENABLE_SYCL "Build with SYCL support" On "ENABLE_SYCL" Off)
cmake_dependent_option(DESUL_ENABLE_TESTS "Build tests" On "ENABLE_TESTS" Off)

include(cmake/blt_boilerplate.cmake)

add_subdirectory(atomics)

install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/desulConfig.cmake"
  DESTINATION lib/cmake/desul)


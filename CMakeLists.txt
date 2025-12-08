cmake_minimum_required (VERSION 3.10)

project(desul
  LANGUAGES CXX C
  VERSION 0.1.0)

# Default to C++17 if not set and enforce the minimum requirement below after
# setting up BLT below
if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++17" CACHE STRING "")
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
         message(FATAL_ERROR "DESUL_${option} set to ON, but ${option} is OFF. Please set ${option} to ON enable this feature.")
      endif ()
   endif ()
endforeach ()


include(CMakeDependentOption)
option(ENABLE_SYCL "Enable SYCL support" OFF) # option does not actually yet exist on the BLT side
cmake_dependent_option(DESUL_ENABLE_CUDA "Build with CUDA support" ON "ENABLE_CUDA" OFF)
cmake_dependent_option(DESUL_ENABLE_HIP "Build with HIP support" ON "ENABLE_HIP" OFF)
cmake_dependent_option(DESUL_ENABLE_OPENMP "Build with OpenMP support" ON "ENABLE_OPENMP" OFF)
cmake_dependent_option(DESUL_ENABLE_SYCL "Build with SYCL support" ON "ENABLE_SYCL" OFF)
cmake_dependent_option(DESUL_ENABLE_TESTS "Build tests" ON "ENABLE_TESTS" OFF)

include(cmake/blt_boilerplate.cmake)
if (CMAKE_CXX_STANDARD LESS 17)
  message(FATAL_ERROR "desul requires a minimum of C++17")
endif()

add_subdirectory(atomics)

install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/desulConfig.cmake"
  DESTINATION lib/cmake/desul)


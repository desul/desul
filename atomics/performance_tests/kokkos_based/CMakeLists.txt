if(DESUL_ENABLE_TESTS)
  find_package(Kokkos 4 REQUIRED)
  blt_add_library(
    NAME
    kokkos_perf_test_main
    SOURCES
    UnitTestMainInit.cpp
    DEPENDS_ON
    desul_atomics
    gtest
    Kokkos::kokkos
    ${DESUL_BACKENDS})

  set(DESUL_PERF_TESTS
      double
      float
      int32_t
      int64_t
      uint32_t
      uint64_t)
  # Device atomic operations on types that are not lock-free (compound types)
  # require the device lock arrays.
  if(DESUL_ENABLE_DEVICE_LOCK_ARRAYS
     OR NOT (DESUL_ENABLE_CUDA
             OR DESUL_ENABLE_HIP
             OR DESUL_ENABLE_SYCL))
    list(APPEND DESUL_PERF_TESTS compound_int32_t_2 compound_uint64_t_3)
  endif()
  foreach(T ${DESUL_PERF_TESTS})
    blt_add_executable(
      NAME
      ${T}_perf_test
      SOURCES
      ${T}_loc.cpp
      DEPENDS_ON
      desul_atomics
      kokkos_perf_test_main
      gtest
      Kokkos::kokkos
      ${DESUL_BACKENDS})
    blt_add_test(NAME ${T}_perf_test COMMAND ${T}_perf_test)
  endforeach()
endif()

if(DESUL_ENABLE_TESTS)
find_package(Kokkos 3 REQUIRED)
add_library(kokkos_perf_test_main
            UnitTestMainInit.cpp)
target_link_libraries(kokkos_perf_test_main desul_atomics gtest Kokkos::kokkos)

foreach(T compound_int32_t_2 compound_uint64_t_3 double float int32_t int64_t uint32_t uint64_t)
  add_executable(${T}_perf_test
                 ${T}_loc.cpp)
  target_link_libraries(${T}_perf_test desul_atomics kokkos_perf_test_main gtest Kokkos::kokkos)
  add_test(NAME ${T}_perf_test COMMAND ${T}_perf_test WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endforeach()
endif()

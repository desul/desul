if(ENABLE_TESTS)
find_package(Kokkos 3 REQUIRED)
blt_add_library(NAME kokkos_test_main
                SOURCES UnitTestMainInit.cpp
                DEPENDS_ON desul_atomics gtest Kokkos::kokkos ${DESUL_BACKENDS})
foreach(T complexdouble  compound  double  int  int64_t)
blt_add_executable(NAME ${T}
                   SOURCES ${T}.cpp
                   DEPENDS_ON desul_atomics kokkos_test_main gtest Kokkos::kokkos ${DESUL_BACKENDS})
blt_add_test(NAME ${T}
             COMMAND ${T})
endforeach()
endif()

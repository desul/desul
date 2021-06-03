if(DESUL_ENABLE_TESTS)
find_package(Kokkos 3 REQUIRED)

# Make sure that Kokkos was built with the requested backends
foreach(BACKEND ${DESUL_BACKENDS})
  string(TOUPPER ${BACKEND} BACKEND)
  kokkos_check(DEVICES ${BACKEND})
endforeach()

blt_add_library(NAME kokkos_unit_test_main
                SOURCES UnitTestMainInit.cpp
                DEPENDS_ON desul_atomics gtest Kokkos::kokkos ${DESUL_BACKENDS})
foreach(T complexdouble  complexfloat  double  float  int  longint  longlongint  unsignedint  unsignedlongint)
blt_add_executable(NAME ${T}_unit_test
                   SOURCES ${T}.cpp
                   DEPENDS_ON desul_atomics kokkos_unit_test_main gtest Kokkos::kokkos ${DESUL_BACKENDS})
blt_add_test(NAME ${T}_unit_test
             COMMAND ${T}_unit_test)
endforeach()
endif()

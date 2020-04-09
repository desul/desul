if(ENABLE_TESTS)
find_package(Kokkos 3 REQUIRED)
blt_add_library(NAME kokkos_test_main
                SOURCES UnitTestMainInit.cpp
                DEPENDS_ON desul_atomics gtest Kokkos::kokkos cuda)
foreach(T complexdouble  complexfloat  double  float  int  longint  longlongint  unsignedint  unsignedlongint)
blt_add_executable(NAME ${T}
                   SOURCES ${T}.cpp
                   DEPENDS_ON desul_atomics kokkos_test_main gtest Kokkos::kokkos cuda)
blt_add_test(NAME ${T}
             COMMAND ${T})
endforeach()
endif()
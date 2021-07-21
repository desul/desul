if(DESUL_ENABLE_TESTS)
find_package(Kokkos 3 REQUIRED)

# Make sure that Kokkos was built with the requested backends
foreach(BACKEND ${DESUL_BACKENDS})
  string(TOUPPER ${BACKEND} BACKEND)
  kokkos_check(DEVICES ${BACKEND})
endforeach()

add_library(kokkos_unit_test_main
            UnitTestMainInit.cpp)
target_link_libraries(kokkos_unit_test_main Kokkos::kokkos gtest desul_atomics)

foreach(T complexdouble  complexfloat  double  float  int  longint  longlongint  unsignedint  unsignedlongint)
  add_executable(${T}_unit_test
               ${T}.cpp)
  target_link_libraries(${T}_unit_test desul_atomics kokkos_unit_test_main gtest Kokkos::kokkoscore)
  add_test(NAME ${T}_unit_test COMMAND ${T}_unit_test WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endforeach()
endif()

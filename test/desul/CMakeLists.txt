set(desul_headers Test.hpp)

set(desul_test_dependencies desul gtest)
blt_list_append(TO desul_test_dependencies ELEMENTS cuda IF ENABLE_CUDA)

blt_add_executable(NAME test_array
                   SOURCES test_array.cpp
                   HEADERS ${desul_headers}
                   DEPENDS_ON ${desul_test_dependencies})

target_include_directories(test_array
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(test_array
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_test(NAME test_array
             COMMAND test_array)



blt_add_executable(NAME test_utility
                   SOURCES test_utility.cpp
                   DEPENDS_ON ${desul_test_dependencies})

target_include_directories(test_utility
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(test_utility
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_test(NAME test_utility
             COMMAND test_utility)


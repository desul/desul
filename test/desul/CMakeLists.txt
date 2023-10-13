set (desul_test_dependencies desul gtest)

blt_add_executable(NAME test_array
                   SOURCES test_array.cpp
                   DEPENDS_ON ${desul_test_dependencies})

target_include_directories(test_array
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(test_array
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_test(NAME test_array
             COMMAND test_array)


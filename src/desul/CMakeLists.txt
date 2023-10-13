set(desul_headers
    Macros.h
    array.hpp
    utility.hpp)

set(desul_depends )
blt_list_append(TO desul_depends ELEMENTS cuda IF ${ENABLE_CUDA})
blt_list_append(TO desul_depends ELEMENTS blt::hip IF ${ENABLE_HIP})

blt_add_library(NAME desul
                HEADERS ${desul_headers}
                DEPENDS_ON ${desul_depends})

target_include_directories(desul INTERFACE ${PROJECT_SOURCE_DIR}/src)

install(TARGETS desul
        EXPORT desul-targets
        DESTINATION lib
        INCLUDES DESTINATION include/desul)

install(FILES ${desul_headers}
        DESTINATION include/desul)

install(EXPORT desul-targets
        NAMESPACE desul
        DESTINATION lib/cmake/desul)


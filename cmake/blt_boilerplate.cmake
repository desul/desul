if (NOT BLT_LOADED)
    if (DEFINED BLT_SOURCE_DIR)
        # Support having a shared BLT outside of the repository if given a BLT_SOURCE_DIR
        if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
            message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
        endif()
    else()
        # Use internal BLT if no BLT_SOURCE_DIR is given
        set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/external/blt" CACHE PATH "")
        if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
            message(FATAL_ERROR
                "The BLT git submodule is not present. "
                "Either run the following two commands in your git repository: \n"
                "    git submodule init\n"
                "    git submodule update\n"
                "Or add -DBLT_SOURCE_DIR=/path/to/blt to your CMake command." )
        endif()
    endif()

    include(${BLT_SOURCE_DIR}/SetupBLT.cmake)
endif()

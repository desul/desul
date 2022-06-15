foreach(component ${desul_FIND_COMPONENTS})
  include(${CMAKE_CURRENT_LIST_DIR}/desul_${component}Config.cmake)
endforeach()

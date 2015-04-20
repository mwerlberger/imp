project(imp_core)
cmake_minimum_required(VERSION 2.8.0)

cmake_policy(SET CMP0054 OLD)

find_package(catkin_simple REQUIRED)
catkin_simple()

include(imp_macros)
imp_setup()


set(HEADERS
  include/imp/core/types.hpp
  include/imp/core/exception.hpp
  include/imp/core/size.hpp
  include/imp/core/roi.hpp
  include/imp/core/pixel.hpp
  include/imp/core/pixel_enums.hpp
  include/imp/core/linearmemory_base.hpp
  include/imp/core/linearmemory.hpp
  include/imp/core/memory_storage.hpp
  )

set(SOURCES
  src/linearmemory.cpp
  )

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME})

###
### GTEST
###

catkin_add_gtest(${PROJECT_NAME}-test
  test/test_main.cpp
  test/size_test.cpp
  test/roi_test.cpp
  test/vec_test.cpp
)
target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME} pthread)

cs_install()
cs_export()


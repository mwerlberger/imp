project(imp_image)
cmake_minimum_required(VERSION 2.8.0)

cmake_policy(SET CMP0054 OLD)

find_package(catkin_simple REQUIRED)
catkin_simple()

include(imp_macros)
imp_setup()

set(HEADERS
  include/imp/image/image_base.hpp
  include/imp/image/image.hpp
  include/imp/image/image_raw.hpp
  include/imp/image/image_defs.hpp
  )

set(SOURCES
  src/image_raw.cpp
  )

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME})

###
### GTEST
###

# catkin_add_gtest(${PROJECT_NAME}-test
# )
# target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME} pthread)

cs_install()
cs_export()


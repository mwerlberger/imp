project(imp_bridge_pangolin)
cmake_minimum_required(VERSION 2.8.0)

cmake_policy(SET CMP0054 OLD)

find_package(catkin_simple REQUIRED)
catkin_simple()

include(imp_macros)
imp_setup()

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x -D__STRICT_ANSI__")

set(HEADERS
  include/imp/bridge/pangolin/imread.hpp
  include/imp/bridge/pangolin/pangolin_display.hpp
  )

set(SOURCES
  src/empty.cpp
  )

set(IMPL_FILES
  )

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS} ${IMPL_FILES})
target_link_libraries(${PROJECT_NAME})

###
### GTESTS
###

# TODO (MWE)

# catkin_add_gtest(${PROJECT_NAME}-test
#    test/test_main.cpp
#    test/todo.cpp
# )
# target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME} pthread)

cs_install()
cs_export()

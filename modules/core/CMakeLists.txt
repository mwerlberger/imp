project(imp_core)
cmake_minimum_required(VERSION 2.8.0)

find_package(catkin_simple REQUIRED)
catkin_simple()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_definitions(-Wall)
add_definitions(-Werror)
add_definitions(-g)
add_definitions(-fPIC)

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x -D__STRICT_ANSI__")

find_package( OpenCV REQUIRED core)
include_directories(${OpenCV_INCLUDE_DIRES})

#include_directories(${Boost_INCLUDE_DIRS})

set(HEADERS
  include/imp/core/types.hpp
  include/imp/core/exception.hpp
  include/imp/core/size.hpp
  include/imp/core/roi.hpp
  include/imp/core/pixel.hpp
  include/imp/core/pixel_enums.hpp
  include/imp/core/linearmemory_base.hpp
  include/imp/core/linearmemory.hpp
  include/imp/core/image_base.hpp
  include/imp/core/image.hpp
  include/imp/core/memory_storage.hpp
  include/imp/core/image_raw.hpp
  include/imp/core/cv_connector_pixel_types.hpp
  include/imp/core/image_cv.hpp
  )

set(SOURCES
  src/linearmemory.cpp
  src/image_raw.cpp
  src/cv_connector_pixel_types.cpp
  src/image_cv.cpp
  )


cs_add_library(${PROJECT_NAME}
   ${SOURCES} ${HEADERS}
   )
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

###
### GTESTS
###

catkin_add_gtest(${PROJECT_NAME}-test
  test/test_main.cpp
  test/size_test.cpp
)
target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME} pthread)

cs_install()
cs_export()


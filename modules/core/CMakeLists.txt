cmake_minimum_required(VERSION 3.0.0)
project(imp_core)


find_package(catkin_simple REQUIRED)
catkin_simple()

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x -D__STRICT_ANSI__")

find_package( OpenCV REQUIRED COMPONENCTS core)
inlcude_directories(${OpenCV_INCLUDE_DIRES})

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

#############
## Testing ##
#############
# # Avoid clash with tr1::tuple: https://code.google.com/p/googletest/source/browse/trunk/README?r=589#257
# add_definitions(-DGTEST_USE_OWN_TR1_TUPLE=0)

catkin_add_gtest(${PROJECT_NAME}-test
  test/test_main.cpp
  test/size_test.cpp
)
target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME} pthread)

cs_install()
# cs_export(CFG_EXTRAS export_flags.cmake)


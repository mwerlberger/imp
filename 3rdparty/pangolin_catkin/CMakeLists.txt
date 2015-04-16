cmake_minimum_required(VERSION 2.8.3)
project(pangolin_catkin)

find_package(catkin_simple REQUIRED)
catkin_simple()

include(ExternalProject)

file(MAKE_DIRECTORY ${CATKIN_DEVEL_PREFIX}/include)

ExternalProject_Add(pangolin_src
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../pangolin
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CATKIN_DEVEL_PREFIX}
    -DCMAKE_BUILD_TYPE:STRING=Release
    -DBUILD_PANGOLIN_OCULUS:BOOL=OFF -DBUILD_EXAMPLES:BOOL=OFF
    -DCPP11_NO_BOOST:BOOL=ON
  BUILD_COMMAND make -j 8
  INSTALL_COMMAND make install -j 8
)

cs_add_library(${PROJECT_NAME} src/empty.cpp)
add_dependencies(${PROJECT_NAME} pangolin_src)

cs_install()
cs_export(INCLUDE_DIRS ${CATKIN_DEVEL_PREFIX}/include)

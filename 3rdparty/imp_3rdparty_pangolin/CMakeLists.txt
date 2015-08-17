cmake_minimum_required(VERSION 2.8.3)
project(imp_3rdparty_pangolin)

find_package(catkin_simple REQUIRED)
catkin_simple()

include(ExternalProject)

file(MAKE_DIRECTORY ${CATKIN_DEVEL_PREFIX}/include)

option(PANGOLIN_USE_GUI "Build 3rdparty pangolin package with GUI support" ON)
option(PANGOLIN_USE_VIDEO "Build 3rdparty pangolin package with Video support" OFF)
option(PANGOLIN_USE_OCULUS "Build 3rdparty pangolin package with Oculus support" OFF)
option(PANGOLIN_USE_VARS "Build 3rdparty pangolin package with Vars support" OFF)

find_package(FREEGLUT QUIET)
find_package(GLUT QUIET)

set(PANGOLIN_CMAKE_OPTIONS "")
if (PANGOLIN_USE_GUI)
   list(APPEND PANGOLIN_CMAKE_OPTIONS "-DBUILD_PANGOLIN_GUI:BOOL=ON")
   find_package(OpenGL REQUIRED QUIET)
   find_package(GLEW REQUIRED QUIET)
else()
   list(APPEND PANGOLIN_CMAKE_OPTIONS "-DBUILD_PANGOLIN_GUI:BOOL=OFF")
endif()

if (PANGOLIN_USE_VIDEO)
   list(APPEND PANGOLIN_CMAKE_OPTIONS "-DBUILD_PANGOLIN_VIDEO:BOOL=ON")
   find_package(DepthSense)
else()
   list(APPEND PANGOLIN_CMAKE_OPTIONS "-DBUILD_PANGOLIN_VIDEO:BOOL=OFF")
endif()

if (PANGOLIN_USE_OCULUS)
   list(APPEND PANGOLIN_CMAKE_OPTIONS "-DBUILD_PANGOLIN_OCULUS:BOOL=ON")
else()
   list(APPEND PANGOLIN_CMAKE_OPTIONS "-DBUILD_PANGOLIN_OCULUS:BOOL=OFF")
endif()

if (PANGOLIN_USE_VARS)
   list(APPEND PANGOLIN_CMAKE_OPTIONS "-DBUILD_PANGOLIN_VARS:BOOL=ON")
else()
   list(APPEND PANGOLIN_CMAKE_OPTIONS "-DBUILD_PANGOLIN_VARS:BOOL=OFF")
endif()

if (CMAKE_BUILD_TYPE)
   list(APPEND PANGOLIN_CMAKE_OPTIONS "-DCMAKE_BUILD_TYPE:String=${CMAKE_BUILD_TYPE}")
endif()

message(STATUS "PANGOLIN_CMAKE_OPTIONS: ${PANGOLIN_CMAKE_OPTIONS}")
ExternalProject_Add(pangolin_src
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../pangolin
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CATKIN_DEVEL_PREFIX}
    -DBUILD_SHARED_LIBS:BOOL=ON
    ${PANGOLIN_CMAKE_OPTIONS}
    -DBUILD_EXAMPLES:BOOL=OFF
    -DCPP11_NO_BOOST:BOOL=ON
  BUILD_COMMAND make install -j 8
  INSTALL_COMMAND make install -j 8
)

message(STATUS "CMAKE_SHARED_LIBRARY_SUFFIX=${CMAKE_SHARED_LIBRARY_SUFFIX}")

cs_add_library(${PROJECT_NAME} src/empty.cpp)
add_dependencies(${PROJECT_NAME} pangolin_src)

cs_install()
cs_export(INCLUDE_DIRS ${CATKIN_DEVEL_PREFIX}/include
   CFG_EXTRAS imp_3rdparty_pangolin-extras.cmake)

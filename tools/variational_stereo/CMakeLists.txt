project(imp_tools_variational_stereo)
cmake_minimum_required(VERSION 2.8.0)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/imp_ext_finders.cmake)
find_cuda()

find_package(catkin_simple REQUIRED)
catkin_simple()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wall -g")

cs_add_executable(variational_stereo_test variational_stereo_test.cpp)
target_link_libraries(variational_stereo_test)

cs_add_executable(variational_epipolar_stereo_test variational_epipolar_stereo_test.cpp)
target_link_libraries(variational_epipolar_stereo_test)


cs_install()
cs_export()

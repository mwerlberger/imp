project(imp_ros_denoising)
cmake_minimum_required(VERSION 2.8.0)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/imp_ext_finders.cmake)
find_cuda()

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wall -g")

#
# binaries
#

cs_add_executable(ros_denoising_rof
   src/rof.cpp
)
target_link_libraries(ros_denoising_rof  ${CUDA_LIBRARIES})


#
# exports
#

cs_install()
cs_export()

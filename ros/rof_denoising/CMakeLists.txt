project(imp_ros_rof_denoising)
cmake_minimum_required(VERSION 2.8.0)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

include(imp_macros)
include(imp_finders)
imp_setup()
imp_find_cuda()

#
# binaries
#

cs_add_executable(rof_node src/rof_node.cpp)
target_link_libraries(rof_node  ${CUDA_LIBRARIES})

#
# exports
#

cs_install()
cs_export()

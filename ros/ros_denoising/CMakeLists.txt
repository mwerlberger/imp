project(imp_ros_denoising)
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
# configs
#
generate_dynamic_reconfigure_options(cfg/rof_node.cfg)
add_dependencies(rof_node ${PROJECT_NAME}_gencfg)


#
# exports
#

cs_install()
cs_export()

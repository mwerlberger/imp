project(imp_ros_rof_denoising)
cmake_minimum_required(VERSION 2.8.0)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_cuda()

include(imp_macros)
imp_setup()

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

project(imp_tools_cmd)
cmake_minimum_required(VERSION 2.8.0)

if(${CMAKE_MAJOR_VERSION} VERSION_GREATER 3.0)
  cmake_policy(SET CMP0054 OLD)
endif(${CMAKE_MAJOR_VERSION} VERSION_GREATER 3.0)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

include(imp_finders)
imp_setup()
imp_find_cuda()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs" )

###
### command line tools
###
cs_cuda_add_executable(texture_issue texture_issue.cpp)
# target_link_libraries(cu_rof_denoising_test)

cs_install()
cs_export()

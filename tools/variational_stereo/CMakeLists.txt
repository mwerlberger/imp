project(imp_tools_variational_stereo)
cmake_minimum_required(VERSION 2.8.0)

find_package(catkin_simple REQUIRED)
catkin_simple()

include(imp_finders)
imp_setup()
imp_find_cuda()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs" )

cs_add_executable(variational_stereo_test variational_stereo_test.cpp)
# target_link_libraries(variational_stereo_test)

cs_add_executable(variational_epipolar_stereo_test variational_epipolar_stereo_test.cpp)
target_link_libraries(variational_epipolar_stereo_test)


cs_install()
cs_export()

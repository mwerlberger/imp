project(imp_cudepth)
cmake_minimum_required(VERSION 2.8.0)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/imp_ext_finders.cmake)
find_cuda()

find_package(catkin_simple REQUIRED)
catkin_simple()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wall -g")

set(HEADERS
  include/imp/cudepth/variational_stereo.hpp
  include/imp/cudepth/variational_stereo_parameters.hpp
  include/imp/cudepth/stereo_solver_enum.hpp
  include/imp/cudepth/stereo_ctf_warping.hpp
  include/imp/cudepth/stereo_ctf_warping.hpp
  include/imp/cudepth/stereo_ctf_warping_level.hpp
  include/imp/cudepth/stereo_ctf_warping_level_huber_l1.cuh
  include/imp/cudepth/stereo_ctf_warping_level_precond_huber_l1.cuh
  include/imp/cudepth/stereo_ctf_warping_level_precond_huber_l1_weighted.cuh
  )

set(SOURCES
  src/variational_stereo.cpp
  src/stereo_ctf_warping.cpp
  )

set(CU_SRCS
  src/cu_k_warped_gradients.cuh

  src/cu_k_stereo_ctf_warping_level_huber_l1.cuh
  src/stereo_ctf_warping_level_huber_l1.cu

  src/cu_k_stereo_ctf_warping_level_precond_huber_l1.cuh
  src/stereo_ctf_warping_level_precond_huber_l1.cu

  src/cu_k_stereo_ctf_warping_level_precond_huber_l1_weighted.cuh
  src/stereo_ctf_warping_level_precond_huber_l1_weighted.cu
  )

cuda_compile(CU_OBJ ${CU_SRCS})

cs_add_library(${PROJECT_NAME}
   ${CU_OBJ} ${SOURCES} ${HEADERS}
   )
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

###
### GTESTS
###

# catkin_add_gtest(${PROJECT_NAME}-test
#   test/test_main.cpp
#   test/size_test.cpp
# )
# target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME} pthread)

cs_install()
cs_export()

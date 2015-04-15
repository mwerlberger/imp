project(imp_cudepth)
cmake_minimum_required(VERSION 2.8.0)

cmake_policy(SET CMP0054 OLD)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/imp_ext_finders.cmake)
find_cuda()

find_package(catkin_simple REQUIRED)
catkin_simple()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wall -g")

set(HEADERS
  include/imp/cudepth/variational_stereo.hpp
  include/imp/cudepth/variational_epipolar_stereo.hpp
  include/imp/cudepth/variational_stereo_parameters.hpp
  include/imp/cudepth/stereo_solver_enum.hpp
  include/imp/cudepth/stereo_ctf_warping.hpp
  include/imp/cudepth/solver_stereo_abstract.hpp
  include/imp/cudepth/stereo_ctf_warping_level_huber_l1.cuh
  include/imp/cudepth/stereo_ctf_warping_level_precond_huber_l1.cuh
  include/imp/cudepth/stereo_ctf_warping_level_precond_huber_l1_weighted.cuh
  # TODO # include/imp/cudepth/solver_epipolar_stereo_abstract.hpp
  include/imp/cudepth/solver_epipolar_stereo_precond_huber_l1.cuh
  )

set(SOURCES
  src/variational_stereo.cpp
  src/variational_epipolar_stereo.cpp
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

  src/k_solver_epipolar_stereo_precond_huber_l1.cuh
  src/solver_epipolar_stereo_precond_huber_l1.cu
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

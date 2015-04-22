project(imp_cu_correspondence)
cmake_minimum_required(VERSION 2.8.0)

cmake_policy(SET CMP0054 OLD)

find_package(catkin_simple REQUIRED)
catkin_simple()

include(imp_macros)
include(imp_finders)
imp_setup()
imp_find_cuda()

set(HEADERS
  include/imp/cu_correspondence/variational_stereo.hpp
  include/imp/cu_correspondence/variational_epipolar_stereo.hpp
  include/imp/cu_correspondence/variational_stereo_parameters.hpp
  include/imp/cu_correspondence/stereo_solver_enum.hpp
  include/imp/cu_correspondence/stereo_ctf_warping.hpp
  include/imp/cu_correspondence/solver_stereo_abstract.hpp
  include/imp/cu_correspondence/solver_stereo_huber_l1.cuh
  include/imp/cu_correspondence/solver_stereo_precond_huber_l1.cuh
  include/imp/cu_correspondence/solver_stereo_precond_huber_l1_weighted.cuh
#  include/imp/cu_correspondence/solver_stereo_phuber_l1_weighted.cuh
  # TODO # include/imp/cu_correspondence/solver_epipolar_stereo_abstract.hpp
  include/imp/cu_correspondence/solver_epipolar_stereo_precond_huber_l1.cuh
#  include/imp/cu_correspondence/occlusion.cuh
  )

set(SOURCES
  src/variational_stereo.cpp
  src/variational_epipolar_stereo.cpp
  src/stereo_ctf_warping.cpp
  )

set(CU_SRCS
  src/warped_gradients_kernel.cuh

  src/solver_stereo_huber_l1_kernel.cuh
  src/solver_stereo_huber_l1.cu

  src/solver_precond_huber_l1_kernel.cuh
  src/solver_stereo_precond_huber_l1.cu

  src/solver_stereo_precond_huber_l1_weighted_kernel.cuh
  src/solver_stereo_precond_huber_l1_weighted.cu

  src/solver_epipolar_stereo_precond_huber_l1_kernel.cuh
  src/solver_epipolar_stereo_precond_huber_l1.cu

#  src/occlusion.cu
  src/occlusion_kernel.cuh
  )

cuda_compile(CU_OBJ ${CU_SRCS})

#foreach(cu_file ${CU_SRCS})
#  message(STATUS "Process file: ${cu_file}")
#  set_source_files_properties(cu_file PROPERTIES HEADER_FILE_ONLY TRUE)
#endforeach()

cs_add_library(${PROJECT_NAME}
   ${CU_OBJ} ${CU_SRCS} ${SOURCES} ${HEADERS}
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

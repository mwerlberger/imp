project(imp_cudepth)
cmake_minimum_required(VERSION 2.8.0)

find_package(catkin_simple REQUIRED)
catkin_simple()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wall -g")

##
# CUDA
##
find_package(CUDA)
add_definitions(-DIMP_WITH_CUDA)
list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo
   -use_fast_math -Xptxas -dlcm=cg -std=c++11)

include_directories(
   include
   ${CUDA_INCLUDE_DIRS}
   ${CUDA_SDK_INCLUDE_DIR}
   ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda_toolkit/${CUDA_VERSION_STRING}/include
   )

# Checking cuda version
if(CUDA_VERSION_STRING STREQUAL "7.0")
   # CUDA 7.0
   #imp_debug("IMP library compiled with CUDA 7.0")
   add_definitions(-DCUDA_VERSION_70)
elseif()
   message(FATAL_ERROR "unknown CUDA version. some things might not be tested.")
endif()


# Selection of compute capability via environment variable
if("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.1")
   list(APPEND CUDA_NVCC_FLAGS -arch=sm_11)
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.2")
   list(APPEND CUDA_NVCC_FLAGS -arch=sm_12)
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.3")
   list(APPEND CUDA_NVCC_FLAGS -arch=sm_13)
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "2.0")
   list(APPEND CUDA_NVCC_FLAGS -arch=sm_20)
   list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "2.1")
   list(APPEND CUDA_NVCC_FLAGS -arch=sm_21)
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "3.0")
   list(APPEND CUDA_NVCC_FLAGS -arch=sm_30)
   list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "3.5")
   list(APPEND CUDA_NVCC_FLAGS -arch=sm_35)
   list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
else()
   list(APPEND CUDA_NVCC_FLAGS -arch=sm_30)
endif()


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

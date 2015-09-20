project(texture_issue_showcase)
cmake_minimum_required(VERSION 2.8.0)

#############################################################################
# compiler flags, etc.
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Werror -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

#############################################################################
# external dependencies
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
   ${CUDA_INCLUDE_DIRS}
   ${CUDA_SDK_INCLUDE_DIR}
   #${CMAKE_CURRENT_SOURCE_DIR}/../../cuda_toolkit/${CUDA_VERSION_STRING}/include
   )

#############################################################################
# cuda setup
list (APPEND CUDA_NVCC_FLAGS -use_fast_math -std=c++11)
list (APPEND CUDA_NVCC_FLAGS --compiler-options;-fno-strict-aliasing;)
list (APPEND CUDA_NVCC_FLAGS --compiler-options;-fPIC;)
list (APPEND CUDA_NVCC_FLAGS --compiler-options;-Wall;)
list (APPEND CUDA_NVCC_FLAGS --compiler-options;-Werror;)
if(CMAKE_BUILD_TYPE MATCHES Debug)
   list (APPEND CUDA_NVCC_FLAGS --device-debug)
   list (APPEND CUDA_NVCC_FLAGS --compiler-options;-g;)
   list (APPEND CUDA_NVCC_FLAGS --compiler-options;-rdynamic;)
   list (APPEND CUDA_NVCC_FLAGS --compiler-options;-lineinfo;)
   list (APPEND CUDA_NVCC_FLAGS --ptxas-options=-v;)
endif()
set(CUDA_SEPARABLE_COMPILATION OFF)
# set to OFF cuda files are added to multiple targets
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(BUILD_SHARED_LIBS ON)


# Checking cuda version
if(CUDA_VERSION_STRING STREQUAL "7.0")
   # CUDA 7.0
   message(STATUS "found CUDA 7.0")
   add_definitions(-DCUDA_VERSION_70)
elseif(CUDA_VERSION_STRING STREQUAL "7.5")
   # CUDA 7.5
   message(STATUS "found CUDA 7.5")
   add_definitions(-DCUDA_VERSION_70)
   add_definitions(-DCUDA_VERSION_75)
else()
   message(FATAL_ERROR "unknown CUDA version. some things might not be tested.")
endif()


# Selection of compute capability via environment variable
if("$ENV{NV_COMPUTE_CAPABILITY}" MATCHES "1.1")
   list(APPEND CUDA_NVCC_FLAGS -arch=sm_11)
elseif("$ENV{NV_COMPUTE_CAPABILITY}" MATCHES "1.2")
   list(APPEND CUDA_NVCC_FLAGS -arch=sm_12)
elseif("$ENV{NV_COMPUTE_CAPABILITY}" MATCHES "1.3")
   list(APPEND CUDA_NVCC_FLAGS -arch=sm_13)
elseif("$ENV{NV_COMPUTE_CAPABILITY}" MATCHES "2.0")
   list(APPEND CUDA_NVCC_FLAGS -arch=sm_20)
   list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
elseif("$ENV{NV_COMPUTE_CAPABILITY}" MATCHES "2.1")
   list(APPEND CUDA_NVCC_FLAGS -arch=sm_21)
elseif("$ENV{NV_COMPUTE_CAPABILITY}" MATCHES "3.0")
   list(APPEND CUDA_NVCC_FLAGS -arch=sm_30)
   list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
elseif("$ENV{NV_COMPUTE_CAPABILITY}" MATCHES "3.5")
   list(APPEND CUDA_NVCC_FLAGS -arch=sm_35)
   list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
else()
   list(APPEND CUDA_NVCC_FLAGS -arch=sm_30)
endif()


add_definitions(-DIMP_WITH_CUDA)
add_definitions(-DIMP_THROW_ON_CUDA_ERROR)

#############################################################################
# executable
cuda_add_executable(texture_issue_showcase
  texture_issue.cpp
  iterative_kernel_calls.cuh
  iterative_kernel_calls.cu
  )
target_link_libraries(texture_issue_showcase ${OpenCV_LIBS} ${CUDA_LIBRARIES})

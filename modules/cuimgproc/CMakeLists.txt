project(imp_cuimgproc)
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
  include/imp/cuimgproc/cu_image_transform.cuh
  include/imp/cuimgproc/cu_image_filter.cuh
  include/imp/cuimgproc/edge_detectors.cuh
  include/imp/cuimgproc/image_pyramid.hpp
  include/imp/cuimgproc/cu_variational_denoising.cuh
  include/imp/cuimgproc/cu_rof_denoising.cuh
  )

set(SOURCES
  src/image_pyramid.cpp
  )

set(CU_SRCS
  src/image_transform/cu_reduce_impl.cu
  src/cu_image_transform.cu
  src/image_filter/cu_median3x3_impl.cu
  src/image_filter/cu_gauss_impl.cu
  src/cu_image_filter.cu
  src/edge_detectors/natural_edges_impl.cu
  src/edge_detectors.cu
  src/cu_variational_denoising.cu
  src/cu_rof_denoising.cu
  )

cuda_compile(CU_OBJ ${CU_SRCS})

cs_add_library(${PROJECT_NAME}
   ${CU_OBJ} ${SOURCES} ${HEADERS}
   )
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

###
### command line tools/tests
###
find_package( OpenCV REQUIRED core highgui imgproc)
include_directories(${OpenCV_INCLUDE_DIRS})

cs_add_executable(cu_rof_denoising_test test/cu_rof_denoising_test.cpp)
target_link_libraries(cu_rof_denoising_test imp_cuimgproc ${OpenCV_LIBS})

cs_add_executable(image_pyramid_test test/image_pyramid_test.cpp)
target_link_libraries(image_pyramid_test ${PROJECT_NAME} ${OpenCV_LIBS})

cs_add_executable(image_filter_median3x3_test test/image_filter_median3x3_test.cpp)
target_link_libraries(image_filter_median3x3_test ${PROJECT_NAME} ${OpenCV_LIBS})

cs_add_executable(image_filter_gauss_test test/image_filter_gauss_test.cpp)
target_link_libraries(image_filter_gauss_test ${PROJECT_NAME} ${OpenCV_LIBS})

cs_add_executable(natural_edges_test test/natural_edges_test.cpp)
target_link_libraries(natural_edges_test ${PROJECT_NAME} ${OpenCV_LIBS})

#cs_add_executable(example_executable src/main.cc)
#target_link_libraries(example_executable example_library)




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

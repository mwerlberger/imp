project(imp_cuimgproc)
cmake_minimum_required(VERSION 2.8.0)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/imp_ext_finders.cmake)
find_cuda()

find_package(catkin_simple REQUIRED)
catkin_simple()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wall -g")

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
  src/image_transform/cu_resample_impl.cu
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

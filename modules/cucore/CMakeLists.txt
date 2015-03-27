project(imp_cucore)
cmake_minimum_required(VERSION 2.8.0)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/imp_ext_finders.cmake)
find_cuda()

find_package(catkin_simple REQUIRED)
catkin_simple()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wall -Werror -g")

add_definitions(-DIMP_THROW_ON_CUDA_ERROR)

set(HEADERS
  include/imp/cucore/cu_utils.hpp
  include/imp/cucore/cu_exception.hpp
  include/imp/cucore/cu_pixel_conversion.hpp
  include/imp/cucore/cu_memory_storage.cuh
  include/imp/cucore/cu_linearmemory.cuh
  include/imp/cucore/cu_gpu_data.cuh
  include/imp/cucore/cu_image_gpu.cuh
  include/imp/cucore/cu_texture.cuh
  include/imp/cucore/cu_math.cuh
  include/imp/cucore/cu_k_setvalue.cuh
  include/imp/cucore/cu_k_derivative.cuh
  include/imp/cucore/cu_matrix.cuh
  )

set(SOURCES
  src/cu_pixel_conversion.cpp
  )

set(CU_SRC
  src/cu_linearmemory.cu
  src/cu_image_gpu.cu
  src/math/cu_min_max_impl.cu
  src/cu_math.cu
  )

cuda_compile(CU_OBJ ${CU_SRC})

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

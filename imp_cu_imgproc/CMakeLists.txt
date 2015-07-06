project(imp_cu_imgproc)
cmake_minimum_required(VERSION 2.8.0)

if(${CMAKE_MAJOR_VERSION} VERSION_GREATER 3.0)
  cmake_policy(SET CMP0054 OLD)
endif(${CMAKE_MAJOR_VERSION} VERSION_GREATER 3.0)

find_package(catkin_simple REQUIRED)
catkin_simple()

include(imp_finders)
imp_setup()
imp_find_cuda()

set(CU_HDRS
  include/imp/cu_imgproc/iterative_kernel_calls.cuh
  )

set(CU_SRCS
  src/iterative_kernel_calls.cu
  )

#cuda_compile(CU_OBJ ${CU_SRCS} ${CU_HDRS})

cs_cuda_add_library(${PROJECT_NAME}
   ${CU_SRCS} ${CU_HDRS}
   )
#target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${catkin_LIBRARIES})

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

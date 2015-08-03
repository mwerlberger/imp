project(imp_test_package)
cmake_minimum_required(VERSION 2.8.0)

if(${CMAKE_MAJOR_VERSION} VERSION_GREATER 3.0)
  cmake_policy(SET CMP0054 OLD)
endif(${CMAKE_MAJOR_VERSION} VERSION_GREATER 3.0)

find_package(catkin_simple REQUIRED)
catkin_simple()

SET(CMAKE_CXX_FLAGS "-Wno-unknown-pragmas")

include(imp_macros)
include(imp_finders)
imp_setup()
imp_find_cuda()

set(HEADERS
  )

set(SOURCES
  )

set(CU_HDRS
  )

set(CU_SRCS
  src/main.cu
  )

set(HEADERS_TEST
include/imp/imp_test_package/kernel_reduce.hpp
include/imp/imp_test_package/helper_cuda.h
  )

set(SOURCES_TEST
  src/test_reduce.cpp
  )

set(CU_HDRS_TEST
  )

set(CU_SRCS_TEST
  src/kernel_reduce.cu
  )

cs_cuda_add_executable(test_executable ${CU_SRCS} ${CU_HDRS} ${SOURCES} ${HEADERS})
cs_cuda_add_executable(test_reduce ${CU_SRCS_TEST} ${CU_HDRS_TEST} ${SOURCES_TEST} ${HEADERS_TEST})

cs_install()
cs_export()

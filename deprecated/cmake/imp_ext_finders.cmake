
macro(imp_debug)
  string(REPLACE ";" " " __msg "${ARGN}")
  message(STATUS "${__msg}")
endmacro()

macro(imp_warn)
  string(REPLACE ";" " " __msg "${ARGN}")
  message(WARNING "${__msg}")
endmacro()

macro(imp_fatal)
  string(REPLACE ";" " " __msg "${ARGN}")
  message(FATAL_ERROR "${__msg}")
endmacro()

macro(imp_include)
   #imp_debug("[MACRO] imp_include( ${ARGN} )")
   include_directories(${ARGN})
   if (IMP_WITH_CUDA AND CUDA_FOUND)
      cuda_include_directories(${ARGN})
   endif()
endmacro()

##------------------------------------------------------------------------------
macro(find_opencv)
  imp_debug("[MACRO] find_opencv(" ${ARGN} ")")
  set (desired_opencv_modules core)
  if (${ARGC} GREATER 0)
    set (desired_opencv_modules ${ARGN})
  endif()

  imp_debug("desired opencv modules: " ${desired_opencv_modules})
  find_package( OpenCV REQUIRED ${desired_opencv_modules})

  if (DEFINED module)
     set(IMP_${module}_LINK_DEPS "${IMP_${module}_LINK_DEPS};${OpenCV_LIBS}" CACHE INTERNAL
        "linkage dependencies for the module ${module}")
  endif()
  set(IMP_LINK_DEPS "${IMP_LINK_DEPS};${OpenCV_LIBS}" CACHE INTERNAL
     "linkage dependencies for imp")
endmacro()

##------------------------------------------------------------------------------
macro(find_cuda)
   imp_debug("[MACRO] find_cuda(" ${ARGN} ")")


##
# CUDA
##
find_package(CUDA ${ARGN})
add_definitions(-DIMP_WITH_CUDA)
list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo
   -use_fast_math -Xptxas -dlcm=cg -std=c++11)

string(REGEX MATCH "/ccache/" SYSTEM_USE_CCACHE "${CMAKE_C_COMPILER}")
if(SYSTEM_USE_CCACHE)
   # nvcc and ccache are not very good friends, hence we set the host compiler
   # for cuda manually.
   # TODO (MWE) find/check the path of the compiler as well as the OS
   set(CUDA_HOST_COMPILER /usr/bin/gcc)
endif()

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_LIST_DIR: ${CMAKE_CURRENT_LIST_DIR}")

include_directories(
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


  # # include the corresponding cuda helpers for the used cuda version
  # imp_include(${IMP_PATH}/cuda_toolkit/${CUDA_VERSION_STRING}/include)

endmacro()

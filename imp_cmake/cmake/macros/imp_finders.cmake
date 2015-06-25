include(imp_macros)

##------------------------------------------------------------------------------
macro(imp_find_cuda)
   imp_debug("[MACRO] find_cuda(" ${ARGN} ")")
##
# CUDA
##
find_package(CUDA)
imp_debug("CUDA_FOUND: ${CUDA_FOUND}")
imp_debug("CUDA_VERSION_MAJOR: ${CUDA_VERSION_MAJOR}")
if ((NOT ${CUDA_FOUND}) OR (${CUDA_VERSION_MAJOR} LESS 7))
   imp_debug("CUDA not found or too old CUDA version (CUDA_VERSION_MAJOR < 7). Skipping this package.")
   return()
endif()
add_definitions(-DIMP_WITH_CUDA)

list (APPEND CUDA_NVCC_FLAGS -relaxed-constexpr -use_fast_math -std=c++11)
list (APPEND CUDA_NVCC_FLAGS --compiler-options;-fno-strict-aliasing;)
list (APPEND CUDA_NVCC_FLAGS --compiler-options;-fPIC;)
list (APPEND CUDA_NVCC_FLAGS --compiler-options;-Wall;)
list (APPEND CUDA_NVCC_FLAGS --compiler-options;-Werror;)
#list (APPEND CUDA_NVCC_FLAGS -Werror)
if(CMAKE_BUILD_TYPE MATCHES Debug)
   list (APPEND CUDA_NVCC_FLAGS --device-debug)
   list (APPEND CUDA_NVCC_FLAGS --compiler-options;-g;)
   list (APPEND CUDA_NVCC_FLAGS --compiler-options;-rdynamic;)
   list (APPEND CUDA_NVCC_FLAGS --compiler-options;-lineinfo;)
   list (APPEND CUDA_NVCC_FLAGS --ptxas-options=-v;)
endif()

#list (APPEND CUDA_NVCC_FLAGS --device-c)
list (APPEND CUDA_NVCC_FLAGS -rdc=true)

set(CUDA_SEPARABLE_COMPILATION ON)
# set to OFF cuda files are added to multiple targets
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

# nvcc and ccache are not very good friends, hence we set the host compiler
# for cuda manually if ccache is enabled.
##! @todo (MWE) find/check the path of the compiler as well as the OS
string(REGEX MATCH "/ccache/" SYSTEM_USE_CCACHE "${CMAKE_C_COMPILER}")
if(SYSTEM_USE_CCACHE)
   if(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/bin/gcc")
      set(CUDA_HOST_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/gcc)
   elseif(EXISTS "/usr/bin/gcc")
      set(CUDA_HOST_COMPILER /usr/bin/gcc)
   endif()
   imp_debug("CUDA_HOST_COMPILER: ${CUDA_HOST_COMPILER}")
endif()

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


  # # include the corresponding cuda helpers for the used cuda version
  # imp_include(${IMP_PATH}/cuda_toolkit/${CUDA_VERSION_STRING}/include)

  # throw an exception if a CUDA error is caught by the error-check
  ##! @todo (MWE) make this configurable
  add_definitions(-DIMP_THROW_ON_CUDA_ERROR)

endmacro()


##------------------------------------------------------------------------------
macro(imp_find_opencv)
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

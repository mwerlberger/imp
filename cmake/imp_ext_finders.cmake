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

  find_package(CUDA)

  add_definitions(-DIMP_WITH_CUDA)

  imp_include(${CUDA_INCLUDE_DIRS} ${CUDA_SDK_INCLUDE_DIR})
  if (DEFINED module)
     set(IMP_${module}_LINK_DEPS "${IMP_${module}_LINK_DEPS};${CUDA_LIBRARIES}" CACHE INTERNAL
        "linkage dependencies for the module ${module}")
  endif()
  set(IMP_LINK_DEPS "${IMP_LINK_DEPS};${CUDA_LIBRARIES}" CACHE INTERNAL
     "linkage dependencies for imp")

  # imp_debug("CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
  # imp_debug("CUDA_SDK_INCLUDE_DIRS: " ${CUDA_SDK_INCLUDE_DIRS})
  # imp_debug("CUDA_LIBRARIES: " ${CUDA_LIBRARIES})

  list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo
     -use_fast_math -Xptxas -dlcm=cg -std=c++11)

  # include the corresponding cuda helpers for the used cuda version
  imp_include(${IMP_PATH}/cuda_toolkit/${CUDA_VERSION_STRING}/include)


  # Checking cuda version
  if(CUDA_VERSION_STRING STREQUAL "7.0")
     # CUDA 7.0
     imp_debug("IMP library compiled with CUDA 7.0")
     add_definitions(-DCUDA_VERSION_70)
  elseif()
     imp_warn("unknown CUDA version. some things might not be tested.")
  endif()

  ##-----------------------------------------------------------------------------
  # Selection of compute capability via environment variable
  if("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.1")
     imp_debug("IMP library compiled with SM 11")
     list(APPEND CUDA_NVCC_FLAGS -arch=sm_11)
  elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.2")
     imp_debug("IMP library compiled with SM 12")
     list(APPEND CUDA_NVCC_FLAGS -arch=sm_12)
  elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.3")
     imp_debug("IMP library compiled with SM 13")
     list(APPEND CUDA_NVCC_FLAGS -arch=sm_13)
  elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "2.0")
     imp_debug("IMP library compiled with SM 20")
     list(APPEND CUDA_NVCC_FLAGS -arch=sm_20)
     list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
  elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "2.1")
     imp_debug("IMP library compiled with SM 21")
     list(APPEND CUDA_NVCC_FLAGS -arch=sm_21)
  elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "3.0")
     imp_debug("IMP library compiled with SM 30")
     list(APPEND CUDA_NVCC_FLAGS -arch=sm_30)
     list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
  elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "3.5")
     imp_debug("IMP library compiled with SM 35")
     list(APPEND CUDA_NVCC_FLAGS -arch=sm_35)
     list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
  else()
     imp_debug("IMP library compiled with SM 30")
     list(APPEND CUDA_NVCC_FLAGS -arch=sm_30)
  endif()


endmacro()

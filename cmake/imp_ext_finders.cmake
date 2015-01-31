##------------------------------------------------------------------------------
macro(find_opencv)
  imp_debug_message("[MACRO] find_opencv(" ${ARGN} ")")
  set (desired_opencv_modules core)
  if (${ARGC} GREATER 0)
    set (desired_opencv_modules ${ARGN})
  endif()

  imp_debug_message("desired opencv modules: " ${desired_opencv_modules})
  find_package( OpenCV REQUIRED ${desired_opencv_modules})

  list(APPEND IMP_${module}_LINK_DEPS "${OpenCV_LIBS}")
  set(IMP_LINK_DEPS "${IMP_LINK_DEPS};${OpenCV_LIBS}" CACHE INTERNAL
      "linkage dependencies for imp")
endmacro()

##------------------------------------------------------------------------------
macro(find_cuda)
  imp_debug_message("[MACRO] find_cuda(" ${ARGN} ")")

  find_package(CUDA)

  cuda_include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_SDK_INCLUDE_DIR})
  include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_SDK_INCLUDE_DIR})
  list(APPEND IMP_${module}_LINK_DEPS "${CUDA_LIBRARIES}")
  set(IMP_LINK_DEPS "${IMP_LINK_DEPS};${CUDA_LIBRARIES}" CACHE INTERNAL
     "linkage dependencies for imp")

  # imp_debug_message("CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
  # imp_debug_message("CUDA_SDK_INCLUDE_DIRS: " ${CUDA_SDK_INCLUDE_DIRS})
  # imp_debug_message("CUDA_LIBRARIES: " ${CUDA_LIBRARIES})


  # Checking cuda version
  # set defines due to some missing functions in cuda 3.1
  if(CUDA_VERSION_STRING STREQUAL "6.0")
     # CUDA 6.0
     message(STATUS "IMP library compiled with CUDA 6.0")
     add_definitions(-DCUDA_VERSION_60)
  elseif(CUDA_VERSION_STRING STREQUAL "6.5")
     # CUDA 6.5
     message(STATUS "IMP library compiled with CUDA 6.5")
     add_definitions(-DCUDA_VERSION_62)
  elseif(CUDA_VERSION_STRING STREQUAL "7.0")
     # CUDA 7.0
     message(STATUS "IMP library compiled with CUDA 7.0")
     add_definitions(-DCUDA_VERSION_70)
     # add c++11 support for cuda>=7
     list(APPEND CUDA_NVCC_FLAGS -std=c++11)
  elseif()
     message(STATUS "unknown CUDA version. some things might not be tested.")
  endif()

  ##-----------------------------------------------------------------------------
  # Selection of compute capability via environment variable
  if("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.1")
     message(STATUS "IMP library compiled with SM 11")
     list(APPEND CUDA_NVCC_FLAGS -arch=sm_11)
  elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.2")
     message(STATUS "IMP library compiled with SM 12")
     list(APPEND CUDA_NVCC_FLAGS -arch=sm_12)
  elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.3")
     message(STATUS "IMP library compiled with SM 13")
     list(APPEND CUDA_NVCC_FLAGS -arch=sm_13)
  elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "2.0")
     message(STATUS "IMP library compiled with SM 20")
     list(APPEND CUDA_NVCC_FLAGS -arch=sm_20)
  elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "2.1")
     message(STATUS "IMP library compiled with SM 21")
     list(APPEND CUDA_NVCC_FLAGS -arch=sm_21)
  elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "3.0")
     message(STATUS "IMP library compiled with SM 30")
     list(APPEND CUDA_NVCC_FLAGS -arch=sm_30)
  else()
     message(STATUS "IMP library compiled with SM 30")
     list(APPEND CUDA_NVCC_FLAGS -arch=sm_30)
  endif()

  message(STATUS "=======CUDA_NVCC_FLAGS=${CUDA_NVCC_FLAGS}")
  
endmacro()

project(impTests CXX C)
find_package(imp)
include_directories("${IMP_INCLUDE_DIRS}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/common")


##-----------------------------------------------------------------------------
## find 3rd-party dependencies (that are desired with the IMP_WITH_*)
## so we only have to check once.
##-----------------------------------------------------------------------------
message(STATUS "IMP_LIBRARIES: ${IMP_LIBRARIES}")
if (${IMP_WITH_OPENCV})
  find_opencv(core highgui)

  message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
endif()

if (${IMP_WITH_CUDA})
  find_cuda()

  if (CUDA_FOUND)
    set(CUDA_NVCC_FLAGS -std=c++11)
    cuda_include_directories("${IMP_INCLUDE_DIRS}")
    cuda_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/common")
  else()
    set(IMP_WITH_CUDA OFF CACHE PATH "IMP library built without CUDA support (CUDA not found)." FORCE)
  endif()
endif()

##-----------------------------------------------------------------------------
add_subdirectory(core)

if (${IMP_WITH_CUDA})
  add_subdirectory(cucore)
endif()

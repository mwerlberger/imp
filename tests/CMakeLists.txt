project(impTests CXX C)
find_package(imp)

##-----------------------------------------------------------------------------
## find 3rd-party dependencies (that are desired with the IMP_WITH_*)
## so we only have to check once.
##-----------------------------------------------------------------------------
if (${IMP_WITH_OPENCV})
  find_opencv(core highgui)
endif()

if (${IMP_WITH_CUDA})
  find_cuda()

  if (CUDA_FOUND)
  else()
    set(IMP_WITH_CUDA OFF CACHE PATH "IMP library built without CUDA support (CUDA not found)." FORCE)
  endif()
endif()


imp_include("${IMP_INCLUDE_DIRS}")
imp_include("${CMAKE_CURRENT_SOURCE_DIR}/common")

##-----------------------------------------------------------------------------
add_subdirectory(core)

if (${IMP_WITH_CUDA})
  add_subdirectory(cucore)
endif()

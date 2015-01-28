if(NOT IMP_MODULES_PATH)
  set(IMP_MODULES_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
  imp_debug_message("IMP_MODULES_PATH: " ${CMAKE_CURRENT_SOURCE_DIR})
endif()

include_directories(${IMP_MODULES_PATH})

## find 3rd-party dependencies (that are desired with the IMP_WITH_*) so we only have to check once.
if (${IMP_WITH_CUDA})
  find_cuda()

  if (CUDA_FOUND)
    set(CUDA_NVCC_FLAGS --compiler-options;-std=c++11)
    cuda_include_directories(${IMP_MODULES_PATH})
  else()
    set(IMP_WITH_CUDA OFF CACHE PATH "IMP library built without CUDA support (CUDA not found)." FORCE)
  endif()
endif()


##------------------------------------------------------------------------------
# TODO (MWE): make this a macro and add all the modules automagically
add_subdirectory(core)
if (${IMP_WITH_CUDA})
   add_subdirectory(cucore)
endif()

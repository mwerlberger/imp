if(NOT IMP_MODULES_PATH)
   set(IMP_MODULES_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
   imp_debug("IMP_MODULES_PATH: " ${CMAKE_CURRENT_SOURCE_DIR})
endif()

## find 3rd-party dependencies (that are desired with the IMP_WITH_*) so we only have to check once.

# cuda (is first due to cuda_include_directories macro)
if (${IMP_WITH_CUDA})
   find_cuda()

   if (NOT CUDA_FOUND)
      set(IMP_WITH_CUDA OFF CACHE PATH "IMP library built without CUDA support (CUDA was not found)." FORCE)
   endif()
endif()

# opencv
if (${IMP_WITH_OPENCV})
   find_opencv()
   if(NOT OpenCV_LIBS)
      set(IMP_WITH_OPENCV OFF CACHE PATH
         "IMP library built without OpenCV support (OpenCV was not found)." FORCE)
   endif()
endif()

# general include
imp_include(${IMP_MODULES_PATH})

##------------------------------------------------------------------------------
# TODO (MWE): make this a macro and add all the modules automagically
add_subdirectory(core)
if (${IMP_WITH_CUDA})
   add_subdirectory(cucore)
   add_subdirectory(cuimgproc)
endif()

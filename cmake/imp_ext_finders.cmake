##------------------------------------------------------------------------------
macro(find_opencv)
   imp_debug_message("[MACRO] find_opencv(" ${ARGN} ")")
   set (desired_opencv_modules core)
   if (${ARGC} GREATER 0)
      set (desired_opencv_modules ${ARGN})
   endif()
   imp_debug_message("desired opencv modules: " ${desired_opencv_modules})


  find_package( OpenCV REQUIRED ${desired_opencv_modules})
  set(IMP_USE_OPENCV TRUE)

  list(APPEND IMP_${module}_LINK_DEPS "${OpenCV_LIBS}")
endmacro()

##------------------------------------------------------------------------------
macro(find_cuda)
   imp_debug_message("[MACRO] find_cuda(" ${ARGN} ")")

   find_package(CUDA)

   cuda_include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_SDK_INCLUDE_DIR})
   include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_SDK_INCLUDE_DIR})
   list(APPEND IMP_${module}_LINK_DEPS "${CUDA_LIBRARIES}")

   imp_debug_message("CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
   imp_debug_message("CUDA_SDK_INCLUDE_DIRS: " ${CUDA_SDK_INCLUDE_DIRS})
   imp_debug_message("CUDA_LIBRARIES: " ${CUDA_LIBRARIES})
endmacro()
macro(find_opencv)
   imp_debug_message("find_opencv(" ${ARGN} ")")
   set (desired_opencv_modules core)
   if (${ARGC} GREATER 0)
      set (desired_opencv_modules ${ARGN})
   endif()
   imp_debug_message("desired opencv modules: " ${desired_opencv_modules})


  find_package( OpenCV REQUIRED ${desired_opencv_modules})
  set(IMP_USE_OPENCV TRUE)
  target_link_libraries(${MODULE} ${OpenCV_LIBS})
endmacro()
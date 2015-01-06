if(NOT IMP_MODULES_PATH)
   set(IMP_MODULES_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# TODO (MWE): make this a macro and add all the modules automagically
add_subdirectory(core)
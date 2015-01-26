# Local variables for each module
#
# name - short name in lower case (i.e. core)
#
#
# USAGE:
#   set(imp_module_description "Brief module description")
#   imp_add_module(name <dependencies>)
#   imp_module_include_directories(<extra include directories>)
#   imp_create_module()
#
# This is heavily inspired by OpenCV's build system. Thank's guys.

macro(imp_debug_message)
  string(REPLACE ";" " " __msg "${ARGN}")
  message(STATUS "${__msg}")
endmacro()

################################################################################
# adds an IMP module (1st call)
macro(imp_add_module _name)
  imp_debug_message("imp_add_module(" ${_name} ${ARGN} ")")
  string(TOLOWER "${_name}" name)
  set(module imp_${name})

  # TODO (MW): do we want to have automated dependencies (two-pass?)
  # TODO (MW): dependency graph?

  # sanity check if there is no description defined
  if(NOT DEFINED imp_module_description)
    set(imp_module_description "IMP's ${name} module")
  endif()
  # sanity check if there is no init for the build option/switch
  if(NOT DEFINED IMP_BUILD_${module}_MODULE_INIT)
    set(IMP_BUILD_${module}_MODULE_INIT ON)
  endif()

  # module details (cached)
  set(IMP_MODULE_${module}_DESCRIPTION "${imp_module_description}" CACHE INTERNAL "Brief description of the ${module} module")
  set(IMP_MODULE_${module}_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "Location of the ${module} module")
  set(IMP_${module}_LINK_DEPS "" CACHE INTERNAL "linkage dependencies of ${module} module")

  # setup linkage type for the current module
  if((NOT DEFINED IMP_MODULE_LINKAGE AND IMP_BUILD_SHARED_LIBS)
      OR (DEFINED IMP_MODULE_LINKAGE AND IMP_MODULE_LINKAGE STREQUAL SHARED))
    set(IMP_MODULE_${module}_LINKAGE SHARED CACHE INTERNAL "module linkage (SHARED|STATIC)")
  else()
    set(IMP_MODULE_${module}_LINKAGE STATIC CACHE INTERNAL "module linkage (SHARED|STATIC)")
  endif()
  imp_debug_message("IMP_MODULE_${module}_LINKAGE:" ${IMP_MODULE_${module}_LINKAGE})

  # option to disable module
  option(IMP_BUILD_${module}_MODULE "IMP built including the ${module} module" ${IMP_BUILD_${module}_MODULE_INIT})

  project(${module})
endmacro()


################################################################################
# creates an IMP module (last call)
macro(imp_create_module)
  imp_debug_message("imp_create_module(" ${ARGN} ")")
  add_library(${module} ${IMP_MODULE_${module}_LINKAGE} ${ARGN})

  # dll api exports (mainly win32)
  if(${IMP_MODULE_${module}_LINKAGE} STREQUAL SHARED)
    # TODO (MW) define version and public headers
    set_target_properties(${module} PROPERTIES
      DEFINE_SYMBOL IMP_API_EXPORTS
      # VERSION ${${PROJECT_NAME}_VERSION}
      # SOVERSION ${${PROJECT_NAME}_SOVERSION}
      # PUBLIC_HEADER "${IU_PUBLIC_HEADERS}"
      )
  endif()

  # include path
  get_filename_component(IMP_MODULE_${module}_INCLUDE_PATH include ABSOLUTE)
  target_include_directories(${module} PUBLIC ${IMP_MODULE_${module}_INCLUDE_PATH})
  # TODO (MW): include dir private or public?

  message(STATUS "TARGET_FILE: $<TARGET_FILE:${module}>")
  get_target_property(FULL_LIBRARY_NAME ${module} LOCATION)
  set(IMP_MODULE_LIBRARIES_LOCATIONS ${FULL_LIBRARY_NAME} CACHE INTERNAL "")
  set(IMP_MODULE_INCLUDE_PATHS ${IMP_MODULE_${module}_INCLUDE_PATH} CACHE INTERNAL "")
  # imp_debug_message("full_library_name:" ${FULL_LIBRARY_NAME})
  # imp_debug_message("IMP_MODULE_${module}_INCLUDE_PATH:" ${IMP_MODULE_${module}_INCLUDE_PATH})

  imp_debug_message("${module} link dependencies: " ${IMP_${module}_LINK_DEPS})
  target_link_libraries(${module} ${IMP_${module}_LINK_DEPS})
endmacro()

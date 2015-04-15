# Local variables for each module
#
# name - short name in lower case (i.e. core)
#
#
# USAGE:
#   set(imp_module_description "Brief module description")
#   imp_add_module(name <dependencies>)
#   imp_create_module()
#
# This is heavily inspired by OpenCV's build system. Thank's guys.

#include(utils.cmake)

################################################################################
# adds an IMP module (1st call)
macro(imp_add_module _name)
  imp_debug("[MACRO] imp_add_module(" ${_name} ${ARGN} ")")
  string(TOLOWER "${_name}" name)
  set(module ${name})

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

  # include path
  get_filename_component(MODULE_INCLUDE_PATH include ABSOLUTE)

  # cached module details
  set(IMP_${module}_DESCRIPTION "${imp_module_description}" CACHE INTERNAL "Brief description of the ${module} module")
  set(IMP_${module}_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "Location of the ${module} module")
  set(IMP_${module}_INCLUDE_PATH "${MODULE_INCLUDE_PATH}" CACHE INTERNAL "Include path of the ${module} module")
  set(IMP_${module}_LINK_DEPS "" CACHE INTERNAL "linkage dependencies of ${module} module")
  set(IMP_${module}_LIBRARY "" CACHE INTERNAL "path to ${module} library")
  set(IMP_${module}_LIBRARIES "" CACHE INTERNAL "All the libraries that are needed if you wanna use the ${module} module")

  # setup linkage type for the current module
  if((NOT DEFINED IMP_LINKAGE AND IMP_BUILD_SHARED_LIBS)
      OR (DEFINED IMP_LINKAGE AND IMP_LINKAGE STREQUAL SHARED))
    set(IMP_${module}_LINKAGE SHARED CACHE INTERNAL "module linkage (SHARED|STATIC)")
  else()
    set(IMP_${module}_LINKAGE STATIC CACHE INTERNAL "module linkage (SHARED|STATIC)")
  endif()
  imp_debug("IMP_${module}_LINKAGE:" ${IMP_${module}_LINKAGE})

  # option to disable module
  option(IMP_BUILD_${module}_MODULE "IMP built including the ${module} module" ${IMP_BUILD_${module}_MODULE_INIT})

  project(${module} CXX C)
endmacro()


################################################################################
# internal common part for creating the module library (no matter if c++ or cuda)
macro(imp_create_module_internal)
  imp_debug("[MACRO] imp_create_module_internal(${module})")
  # dll api exports (mainly win32)
  if(${IMP_${module}_LINKAGE} STREQUAL SHARED)
    # TODO (MW) define version and public headers
    set_target_properties(${module} PROPERTIES
      DEFINE_SYMBOL IMP_API_EXPORTS
      # VERSION ${${PROJECT_NAME}_VERSION}
      # SOVERSION ${${PROJECT_NAME}_SOVERSION}
      # PUBLIC_HEADER "${IU_PUBLIC_HEADERS}"
      )
  endif()

  get_target_property(LIBRARY_ABS_PATH ${module} LOCATION)
  ## TODO(MWE) we have to get rid of LOCATION property sooner or later....
  #  message(STATUS "TARGET_FILE: $<TARGET_FILE:${module}>")

  # far from optimal but list append doesn't work with cache variables
  set(IMP_${module}_LIBRARY "${LIBRARY_ABS_PATH}" CACHE INTERNAL "path to ${module} library")
  set(IMP_${module}_LIBRARIES "${LIBRARY_ABS_PATH};${IMP_${module}_LINK_DEPS}" CACHE INTERNAL
     "All the libraries that are needed if you wanna use the ${module} module")

  set(IMP_MODULE_LIBRARIES_LOCATIONS "${IMP_MODULE_LIBRARIES_LOCATIONS};${LIBRARY_ABS_PATH}" CACHE INTERNAL "List of the absolute locations of all built module libraries")
  set(IMP_MODULE_INCLUDE_PATHS "${IMP_MODULE_INCLUDE_PATHS};${IMP_${module}_INCLUDE_PATH}" CACHE INTERNAL "List of all module's include paths")

  # include path (TODO private or public?? difference?)
  target_include_directories(${module} PRIVATE ${IMP_${module}_INCLUDE_PATH})

  imp_debug("${module} link dependencies: " ${IMP_${module}_LINK_DEPS})
  target_link_libraries(${module} ${IMP_${module}_LINK_DEPS})
endmacro()

################################################################################
# creates an IMP module (last call)
macro(imp_create_module)
  imp_debug("[MACRO]  imp_create_module(" ${ARGN} ")")
  imp_include(${IMP_MODULE_INCLUDE_PATHS})
  add_library(${module} ${IMP_${module}_LINKAGE} ${ARGN})
  imp_create_module_internal()
endmacro()

################################################################################
# creates an IMP cuda module (last call)
macro(imp_create_cuda_module)
  imp_debug("[MACRO]  imp_create_cuda_module(" ${ARGN} ")")
  imp_include(${IMP_MODULE_INCLUDE_PATHS})
  cuda_add_library(${module} ${IMP_${module}_LINKAGE} ${ARGN})
  imp_create_module_internal()
endmacro()


################################################################################
# adds include paths if we use another mother internally
# @todo (MWE) at the moment we don't have a dependency hirarchy -> check yourself!
macro(imp_include_module MODULE_DEP)
  imp_debug("[MACRO] imp_include_module(${MODULE_DEP}) [module=${module}]")

  imp_include(${IMP_${MODULE_DEP}_INCLUDE_PATH})
  if (DEFINED module)
     list(APPEND IMP_${module}_LINK_DEPS ${IMP_${MODULE_DEP}_LIBRARIES})
  endif()
endmacro()

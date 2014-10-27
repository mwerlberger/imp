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
  if(NOT DEFINED BUILD_${module}_MODULE_INIT)
    set(BUILD_${module}_MODULE_INIT ON)
  endif()

  # module details (cached)
  set(IMP_MODULE_${module}_DESCRIPTION "${imp_module_description}" CACHE INTERNAL "Brief description of the ${module} module")
  #set(IMP_MODULE_${module}_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "Location of the ${module} module")
  #set(IMP_${module}_LINK_DEPS "" CACHE INTERNAL "linkage dependencies") # TODO (MW)

  # option to disable module
  option(BUILD_${module}_MODULE "IMP built including the ${module} module" ${BUILD_${module}_MODULE_INIT})

  # TODO (MW) parse dependencies and generate link string

  
  project(${module})
endmacro()


################################################################################
# creates an IMP module (last call)
macro(imp_create_module)
  imp_debug_message("imp_create_module(" ${ARGN} ")")
  add_library(${module} ${ARGN})

  # include path
  get_filename_component(IMP_MODULE_${module}_INCLUDE_PATH include ABSOLUTE)
  target_include_directories(${module} PUBLIC ${IMP_MODULE_${module}_INCLUDE_PATH})
  # TODO (MW): include dir private or public?
endmacro()

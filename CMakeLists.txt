set(PROJECT_NAME imp)
#set(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWERCASE)
#string(TOUPPER ${PROJECT_NAME} PROJECT_NAME_VAR)
project(${PROJECT_NAME} CXX C)
#set(CMAKE_BUILD_TYPE Debug)
cmake_minimum_required(VERSION 2.8.0)

## internal macros/includes
#include(cmake/utils.cmake)
include(cmake/imp_module.cmake)

# cleanup
set(IMP_MODULE_LIBRARIES_LOCATIONS "" CACHE INTERNAL "List of the absolute locations of all built module libraries")
set(IMP_MODULE_INCLUDE_PATHS "" CACHE INTERNAL "List of all module's include paths")
# set(${IU_PUBLIC_HEADERS} "")

# # general settings
# set( CMAKE_DEBUG_POSTFIX "d" )

# #FILE(TO_CMAKE_PATH "$ENV{CMAKE_MODULE_PATH}" OT_CMAKE_MODULE_PATH)
# FILE(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake" OT_IU_CMAKE_MODULE_PATH)
# SET(CMAKE_MODULE_PATH ${OT_IU_CMAKE_MODULE_PATH})

# convert env variable to internal path to circumvent win32 backslash problems
#FILE(TO_CMAKE_PATH "$ENV{VMLIBRARIES_ROOT}" FL_VMLIBRARIES_ROOT)

##-----------------------------------------------------------------------------
## Current version number:
##-----------------------------------------------------------------------------
set(${PROJECT_NAME}_VERSION "2.0.0")

string(REGEX MATCHALL "[0-9]" ${PROJECT_NAME}_VERSION_PARTS "${${PROJECT_NAME}_VERSION}")

list(GET ${PROJECT_NAME}_VERSION_PARTS 0 ${PROJECT_NAME}_VERSION_MAJOR)
list(GET ${PROJECT_NAME}_VERSION_PARTS 1 ${PROJECT_NAME}_VERSION_MINOR)
list(GET ${PROJECT_NAME}_VERSION_PARTS 2 ${PROJECT_NAME}_VERSION_PATCH)
set(${PROJECT_NAME}_SOVERSION "${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}")

if(WIN32)
    # Postfix of DLLs:
    set(${PROJECT_NAME}_DLLVERSION "${${PROJECT_NAME}_VERSION_MAJOR}${${PROJECT_NAME}_VERSION_MINOR}${${PROJECT_NAME}_VERSION_PATCH}")
    set(${PROJECT_NAME}_DEBUG_POSTFIX d)
else()
    # Postfix of so's:
    #set(${PROJECT_NAME}_DLLVERSION "${${PROJECT_NAME}_VERSION_MAJOR}${${PROJECT_NAME}_VERSION_MINOR}${${PROJECT_NAME}_VERSION_PATCH}")
    set(${PROJECT_NAME}_DLLVERSION "")
    set(${PROJECT_NAME}_DEBUG_POSTFIX)
endif()

# ##-----------------------------------------------------------------------------
# ## Flags if modules are used.
# ## The modules are included per default unless dependencies not found or omitted by the user.
# ## (the core module is always included!)
# OPTION(${PROJECT_NAME}_USE_SPARSE "Including Sparse Matrix module." OFF)
# OPTION(${PROJECT_NAME}_USE_IPP "Including IPP module." ON)
# OPTION(${PROJECT_NAME}_USE_MATLAB "Including Matlab module." ON)
# OPTION(${PROJECT_NAME}_USE_GUI "Including GUI module." OFF)
# OPTION(${PROJECT_NAME}_USE_IO "Including IO module." ON)
# OPTION(${PROJECT_NAME}_USE_IOPGM "Including IOPGM module." ON)
# #OPTION(${PROJECT_NAME}_USE_VIDEOCAPTURE "Including VideCapture IO module." ON)
# #OPTION(${PROJECT_NAME}_USE_PGRCAMERA "Including PointGray IO module." ON)
# OPTION(${PROJECT_NAME}_USE_VIDEOSOURCE "Including Videosource IO module." ON)


##-----------------------------------------------------------------------------
## Compiler Flags
if(UNIX)
  add_definitions(-Wall)
  add_definitions(-Werror)
  add_definitions(-g)
  add_definitions(-fPIC)
endif()

IF(WIN32)
  ADD_DEFINITIONS(-DWIN32)
  # for msvc 8 deprecated warning removal
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE)
  ADD_DEFINITIONS(-D_CRT_NONSTDC_NO_DEPRECATE)
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE_GLOBALS)

  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT.lib")
endif(WIN32)

##-----------------------------------------------------------------------------
## settings / options
option(IMP_BUILD_SHARED_LIBS "Build shared libraries if ON (default) or static ones if OFF." ON)
imp_debug_message("IMP_BUILD_SHARED_LIBS:" ${IMP_BUILD_SHARED_LIBS})

##-----------------------------------------------------------------------------
## include modules
add_subdirectory(core)

#
#
imp_debug_message("************************************************************")
imp_debug_message("IMP BUILD SETUP:")
imp_debug_message("")
imp_debug_message("IMP_MODULE_LIBRARIES_LOCATIONS: ${IMP_MODULE_LIBRARIES_LOCATIONS}")
imp_debug_message("IMP_MODULE_INCLUDE_PATHS: ${IMP_MODULE_INCLUDE_PATHS}")
imp_debug_message("************************************************************")

#
#

##-----------------------------------------------------------------------------
## impConfig.cmake creation for finding IMP
# Create the opengvConfig.cmake file for other cmake projects.
#get_target_property(FULL_LIBRARY_NAME ${PROJECT_NAME} LOCATION)
set(imp_LIBRARIES ${IMP_MODULE_LIBRARIES_LOCATIONS})
set(imp_LIBRARY_DIR ${PROJECT_BINARY_DIR} )
set(imp_INCLUDE_DIR ${IMP_MODULE_INCLUDE_PATHS})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/impConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/impConfig.cmake @ONLY IMMEDIATE)
message(STATUS "PROJECT_NAME=${PROJECT_NAME}")
#export(PACKAGE imp)
export(PACKAGE ${PROJECT_NAME})
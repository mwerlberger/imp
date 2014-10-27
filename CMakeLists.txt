set(PROJECT_NAME IMP)
set(TOLOWER $PROJECT_NAME PROJECT_NAME_LOWERCASE)
#string(TOUPPER ${PROJECT_NAME} PROJECT_NAME_VAR)
project(${PROJECT_NAME} CXX C)
#set(CMAKE_BUILD_TYPE Debug)
cmake_minimum_required(VERSION 2.8.0)

## internal macros/includes
#include(cmake/utils.cmake)
include(cmake/imp_module.cmake)

# # cleanup
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
## include modules
add_subdirectory(core)



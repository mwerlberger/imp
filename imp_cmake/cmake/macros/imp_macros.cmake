macro(imp_debug)
  string(REPLACE ";" " " __msg "${ARGN}")
  message(STATUS "${__msg}")
endmacro()

macro(imp_warn)
  string(REPLACE ";" " " __msg "${ARGN}")
  message(WARNING "${__msg}")
endmacro()

macro(imp_fatal)
  string(REPLACE ";" " " __msg "${ARGN}")
  message(FATAL_ERROR "${__msg}")
endmacro()

macro(imp_include)
   #imp_debug("[MACRO] imp_include( ${ARGN} )")
   include_directories(${ARGN})
   if (IMP_WITH_CUDA AND CUDA_FOUND)
      cuda_include_directories(${ARGN})
   endif()
endmacro()

##! @todo (MWE) extend with default setup stuff that we want to re-use in every package
macro(imp_setup)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -fPIC -Wall")
   ## TODO add  -Werror again
   if (CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
   endif()
endmacro()

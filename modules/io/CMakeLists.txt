set(imp_module_description "IO module")

if (NOT CUDA_FOUND)
  # if cuda is not available we throw an error as this module is useless without cuda.
  # we should never get here though as we already check before...
  imp_fatal("CUDA not available or not correctly configured. CUDA modules can't be included/built. Try to build without CUDA.")
endif()

imp_add_module(io)

# internal module dependencies
imp_include_module(core)
#imp_include_module(cucore)

include_directories(${IMP_${module}_INCLUDE_PATH})
#cuda_include_directories(${IMP_${module}_INCLUDE_PATH})

set(HEADERS
  include/imp/io/opencv_bridge.hpp
  )

set(SOURCES
  #src/opencv_bridge.cpp
  )

set(IMPL_FILES
  )

imp_create_cuda_module(${SOURCES} ${HEADERS} ${IMPL_FILES})

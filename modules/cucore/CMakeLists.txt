set(imp_module_description "CUDA Core Module")

if (NOT CUDA_FOUND)
  # if cuda is not available we throw an error as this module is useless without cuda.
  # we should never get here though as we already check before...
  imp_fatal("CUDA not available or not correctly configured. CUDA modules can't be included/built. Try to build without CUDA.")
endif()

imp_add_module(cucore)

include_directories(${IMP_MODULE_core_INCLUDE_PATH})
cuda_include_directories(${IMP_MODULE_core_INCLUDE_PATH})
include_directories(${IMP_MODULE_${module}_INCLUDE_PATH})
cuda_include_directories(${IMP_MODULE_${module}_INCLUDE_PATH})

set(HEADERS
  include/imp/cucore/cu_memory_storage.cuh
  include/imp/cucore/cu_linearmemory.cuh
#  include/imp/cucore/cu_image_raw.cuh
  )

set(SOURCES
  src/cu_linearmemory.cu
#  src/cu_image_raw.cu
  )

#imp_module_include_directories(${module})
imp_create_cuda_module(${SOURCES} ${HEADERS})

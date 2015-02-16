set(imp_module_description "CUDA Core Module")

if (NOT CUDA_FOUND)
  # if cuda is not available we throw an error as this module is useless without cuda.
  # we should never get here though as we already check before...
  imp_fatal("CUDA not available or not correctly configured. CUDA modules can't be included/built. Try to build without CUDA.")
endif()

imp_add_module(cucore)

# internal module dependencies
imp_include_module(core)

include_directories(${IMP_${module}_INCLUDE_PATH})
cuda_include_directories(${IMP_${module}_INCLUDE_PATH})

set(HEADERS
  include/imp/cucore/cu_utils.hpp
  include/imp/cucore/cu_exception.hpp
  include/imp/cucore/cu_pixel_conversion.hpp
  include/imp/cucore/cu_memory_storage.cuh
  include/imp/cucore/cu_linearmemory.cuh
  include/imp/cucore/cu_gpu_data.cuh
  include/imp/cucore/cu_image_gpu.cuh
  include/imp/cucore/cu_texture.cuh
  include/imp/cucore/cu_math.cuh
  include/imp/cucore/cu_k_setvalue.cuh
  include/imp/cucore/cu_k_derivative.cuh
  )

set(SOURCES
  src/cu_linearmemory.cu
  src/cu_image_gpu.cu
  src/cu_pixel_conversion.cpp
  src/cu_math.cu
  )


set(IMPL_FILES
  src/math/cu_min_max_impl.cu
  )

imp_create_cuda_module(${SOURCES} ${HEADERS} ${IMPL_FILES})

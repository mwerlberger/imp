set(imp_module_description "CUDA Image Processing Module")

imp_add_module(cuimgproc)

# internal module dependencies
imp_include_module(core)
imp_include_module(cucore)

imp_include(${IMP_${module}_INCLUDE_PATH})

set(HEADERS
  include/imp/cuimgproc/cu_image_transform.cuh
  include/imp/cuimgproc/cu_image_filter.cuh
  include/imp/cuimgproc/image_pyramid.hpp
  include/imp/cuimgproc/cu_variational_denoising.cuh
  include/imp/cuimgproc/cu_rof_denoising.cuh
  )

set(SOURCES
  src/cu_image_transform.cu
  src/cu_image_filter.cu
  src/image_pyramid.cpp
  src/cu_variational_denoising.cu
  src/cu_rof_denoising.cu
  )

set(IMPL_FILES
  src/image_transform/cu_reduce_impl.cu
  src/image_filter/cu_median3x3_impl.cu
  src/image_filter/cu_gauss_impl.cu
  )

imp_create_cuda_module(${SOURCES} ${HEADERS} ${IMPL_FILES})

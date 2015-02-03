set(imp_module_description "CUDA Image Processing Module")

imp_add_module(cuimgproc)

# internal module dependencies
imp_include_module(core)
imp_include_module(cucore)

include_directories(${IMP_${module}_INCLUDE_PATH})
cuda_include_directories(${IMP_${module}_INCLUDE_PATH})

set(HEADERS
   include/imp/cuimgproc/cu_variational_denoising.cuh
   include/imp/cuimgproc/cu_rof_denoising.cuh
  )

set(SOURCES
   src/cu_rof_denoising.cu
  )

imp_create_cuda_module(${SOURCES} ${HEADERS})

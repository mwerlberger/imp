set(imp_module_description "IMP Depth Estimation Module (CUDA)")

imp_add_module(cudepth)

# internal module dependencies
imp_include_module(core)
imp_include_module(cucore)
imp_include_module(cuimgproc)

imp_include(${IMP_${module}_INCLUDE_PATH})

set(HEADERS
  include/imp/cudepth/variational_stereo.hpp
  include/imp/cudepth/variational_stereo_parameters.hpp
  include/imp/cudepth/stereo_solver_enum.hpp
  include/imp/cudepth/stereo_ctf_warping.hpp
  include/imp/cudepth/stereo_ctf_warping.hpp
  include/imp/cudepth/stereo_ctf_warping_level.hpp
  include/imp/cudepth/stereo_ctf_warping_level_huber.cuh
  )

set(SOURCES
  src/variational_stereo.cpp
  src/stereo_ctf_warping.cpp
  src/stereo_ctf_warping_level_huber.cu
  )

set(IMPL_FILES
  src/cu_k_warped_gradients.cuh
  src/cu_k_stereo_ctf_warping_level_huber.cuh
  )

imp_create_cuda_module(${SOURCES} ${HEADERS} ${IMPL_FILES})

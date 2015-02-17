set(imp_module_description "IMP Depth Estimation Module (CUDA)")

imp_add_module(cudepth)

# internal module dependencies
imp_include_module(core)
imp_include_module(cucore)
imp_include_module(cuimgproc)

imp_include(${IMP_${module}_INCLUDE_PATH})

set(HEADERS
  include/imp/cudepth/variational_stereo.hpp
  include/imp/cudepth/stereo_solver_enum.hpp
  )

set(SOURCES
  src/variational_stereo.cpp
  )

set(IMPL_FILES
  #src/stereo/
  )

imp_create_cuda_module(${SOURCES} ${HEADERS} ${IMPL_FILES})

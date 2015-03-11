#include <imp/cuimgproc/edge_detectors.cuh>

#include <cstdint>
#include <cuda_runtime.h>

#include <imp/core/types.hpp>
#include <imp/core/pixel.hpp>
#include <imp/core/roi.hpp>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cucore/cu_utils.hpp>
#include <imp/cucore/cu_texture.cuh>

// IMPLEMENTATIONS

// natural image edges
#include "edge_detectors/natural_edges_impl.cu"

namespace imp {
namespace cu {

// intentionally left blank for now
// if you have small helper functions that fit here, feel free to add them.

} // namespace cu
} // namespace imp


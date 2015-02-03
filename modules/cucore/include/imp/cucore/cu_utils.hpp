#ifndef IMP_CU_UTILS_CUH
#define IMP_CU_UTILS_CUH

#include <cuda_runtime_api.h>
#include <cstdint>

namespace imp { namespace cu {

/** Integer division rounding up to next higher integer
 * @param a Numerator
 * @param b Denominator
 * @return a / b rounded up
 */
__device__ __host__ __forceinline__
std::uint32_t divUp(std::uint32_t a, std::uint32_t b)
{
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

} // namespace cu
} // namespace imp

#endif // IMP_CU_UTILS_CUH


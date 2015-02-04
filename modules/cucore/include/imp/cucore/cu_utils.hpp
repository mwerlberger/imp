#ifndef IMP_CU_UTILS_CUH
#define IMP_CU_UTILS_CUH

#include <cuda_runtime_api.h>
#include <cstdint>
#include <imp/core/size.hpp>
#include <imp/core/roi.hpp>

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

template <std::uint16_t _block_size=16>
struct Fragmentation
{
  const std::uint16_t block_size = _block_size;
  imp::Size2u size;
  imp::Roi2u roi;
  dim3 dimBlock;
  dim3 dimGrid;


  Fragmentation() = delete;

  Fragmentation(imp::Size2u sz)
    : size(sz.width(), sz.height())
    , roi(0,0,sz.width(),sz.height())
    , dimBlock(block_size, block_size)
    , dimGrid(divUp(sz.width(), block_size), divUp(sz.height(), block_size))
  {
  }

  Fragmentation(std::uint32_t width, std::uint32_t height)
    : size(width, height)
    , roi(0,0,width,height)
    , dimBlock(block_size, block_size)
    , dimGrid(divUp(width, block_size), divUp(height, block_size))
  {
  }
};

} // namespace cu
} // namespace imp

#endif // IMP_CU_UTILS_CUH


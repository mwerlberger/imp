#ifndef IMP_CU_UTILS_CUH
#define IMP_CU_UTILS_CUH

#include <cuda_runtime_api.h>
#include <cstdint>

#include <imp/core/size.hpp>
#include <imp/core/roi.hpp>
#include <imp/cucore/cu_exception.hpp>

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

//##############################################################################

/** Check for CUDA error */
static inline void checkCudaErrorState(const char* file, const char* function,
                                       const int line)
{
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if( err != cudaSuccess )
    throw imp::cu::Exception("error state check", err, file, function, line);
}

/** Macro for checking on cuda errors
 * @note This check is only enabled when the compile time flag is set
 */
#ifdef IMP_THROW_ON_CUDA_ERROR
  #define IMP_CUDA_CHECK() checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__)
#else
  #define IMP_CUDA_CHECK() do{}while(0)
#endif

static inline float getTotalGPUMemory()
{
  size_t total = 0;
  size_t free = 0;
  cudaMemGetInfo(&free, &total);
  return total/(1024.0f*1024.0f);   // return value in Megabytes
}

static inline float getFreeGPUMemory()
{
  size_t total = 0;
  size_t free = 0;
  cudaMemGetInfo(&free, &total);
  return free/(1024.0f*1024.0f);   // return value in Megabytes
}

static inline void printGPUMemoryUsage()
{
  float total = imp::cu::getTotalGPUMemory();
  float free = imp::cu::getFreeGPUMemory();

  printf("GPU memory usage\n");
  printf("----------------\n");
  printf("   Total memory: %.2f MiB\n", total);
  printf("   Used memory:  %.2f MiB\n", total-free);
  printf("   Free memory:  %.2f MiB\n", free);
}

/** @} */ // end of Error Handling

/** @} */ // end of Cuda Utilities


} // namespace cu
} // namespace imp

#endif // IMP_CU_UTILS_CUH


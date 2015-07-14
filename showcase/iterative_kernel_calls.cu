#include "iterative_kernel_calls.cuh"

#include <iostream>
#include <cuda_runtime.h>

namespace imp {
namespace cu {

//##############################################################################
// HELPER FUNCTIONS

//------------------------------------------------------------------------------
/** Integer division rounding up to next higher integer
 * @param a Numerator
 * @param b Denominator
 * @return a / b rounded up
 */
__host__ __device__ __forceinline__
std::uint32_t divUp(std::uint32_t a, std::uint32_t b)
{
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

//------------------------------------------------------------------------------
template <std::uint16_t block_size_x=32,
          std::uint16_t block_size_y=32,
          std::uint16_t block_size_z=1>
struct Fragmentation
{
  //  imp::Size2u size;
  //  imp::Roi2u roi;
  dim3 dimBlock = dim3(block_size_x, block_size_y, block_size_z);
  dim3 dimGrid;


  Fragmentation() = delete;

  Fragmentation(size_t length)
    : dimGrid(divUp(length, dimBlock.x), dimBlock.x, dimBlock.y)
  {
  }

  Fragmentation(std::uint32_t width, std::uint32_t height)
    : dimGrid(divUp(width, dimBlock.x), divUp(height, dimBlock.y))
  {
  }

  Fragmentation(dim3 _dimGrid, dim3 _dimBlock)
    : dimGrid(_dimGrid)
    , dimBlock(_dimBlock)
  {
  }
};

//-----------------------------------------------------------------------------
template<typename T>
__device__ __forceinline__
T tex2DFetch(
    const Texture2D& tex, float x, float y,
    float mul_x=1.f, float mul_y=1.f, float add_x=0.f, float add_y=0.f)
{
  return ::tex2D<T>(tex.tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
}

//------------------------------------------------------------------------------
static inline void checkCudaErrorState(const char* file, const char* function,
                                       const int line)
{
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if( err != ::cudaSuccess )
    throw Exception("error state check", err, file, function, line);
}

//------------------------------------------------------------------------------
#ifdef IMP_THROW_ON_CUDA_ERROR
#  define IMP_CUDA_CHECK() checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__)
#else
#  define IMP_CUDA_CHECK() cudaDeviceSynchronize()
#endif

//#############################################################################
// KERNELS

//-----------------------------------------------------------------------------
__global__ void kernelCall(float* out_buffer, size_t stride,
                           Texture2D in_tex,
                           size_t width, size_t height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x<width && y<height)
  {
    float val = tex2DFetch<float>(in_tex, x, y);
    out_buffer[y*stride + x] = val;
  }
}

//#############################################################################
// IMPLEMENTATION

//-----------------------------------------------------------------------------
IterativeKernelCalls::IterativeKernelCalls()
  : in_tex_(nullptr)
{
}

//-----------------------------------------------------------------------------
IterativeKernelCalls::~IterativeKernelCalls()
{

}

//-----------------------------------------------------------------------------
void IterativeKernelCalls::init()
{
  // setup textures
  in_tex_.reset(new Texture2D(
                  in_buffer_, pitch_, cudaCreateChannelDesc<float>(), width_, height_,
                  false, cudaFilterModeLinear, cudaAddressModeClamp, cudaReadModeElementType));
  IMP_CUDA_CHECK();

  // init internal vars
  Fragmentation<> fragmentation(width_, height_);
  kernelCall
      <<< fragmentation.dimGrid, fragmentation.dimBlock
      >>> (out_buffer_, pitch_/sizeof(float), *in_tex_, width_, height_);
  IMP_CUDA_CHECK();


  cudaMallocPitch((void**)&unrelated_, &unrelated_pitch_, width_*sizeof(float), height_);

  IMP_CUDA_CHECK();
}

//-----------------------------------------------------------------------------
void IterativeKernelCalls::run(float* dst, const float* src, size_t pitch,
                               std::uint32_t width, std::uint32_t height,
                               bool break_things)
{
  in_buffer_ = src;
  out_buffer_ = dst;
  width_ = width;
  height_ = height;
  pitch_ = pitch;

  this->init();

  Fragmentation<> fragmentation(width, height);

  // if this is called textured are messed up
  if (break_things)
  {
    this->breakThings();
  }

  // dummy copy kernel
  kernelCall
      <<< fragmentation.dimGrid, fragmentation.dimBlock
      >>> (out_buffer_, pitch_/sizeof(float), *in_tex_, width_, height_);
  IMP_CUDA_CHECK();
}

//-----------------------------------------------------------------------------
void IterativeKernelCalls::breakThings()
{
  std::cout << "!!!! breaking things" << std::endl;


  cudaTextureObject_t tex_object;

  cudaResourceDesc tex_res;
  std::memset(&tex_res, 0, sizeof(tex_res));
  tex_res.resType = cudaResourceTypePitch2D;
  tex_res.res.pitch2D.width = width_;
  tex_res.res.pitch2D.height = height_;
  tex_res.res.pitch2D.pitchInBytes = unrelated_pitch_;
  tex_res.res.pitch2D.devPtr = (void*)unrelated_;
  tex_res.res.pitch2D.desc = cudaCreateChannelDesc<float>();

  cudaTextureDesc tex_desc;
  std::memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.normalizedCoords = 0;
  tex_desc.filterMode = cudaFilterModeLinear;
  tex_desc.addressMode[0] = cudaAddressModeClamp;
  tex_desc.addressMode[1] = cudaAddressModeClamp;
  tex_desc.readMode = cudaReadModeElementType;

  cudaError_t err = cudaCreateTextureObject(&tex_object, &tex_res, &tex_desc, 0);
  if  (err != ::cudaSuccess)
  {
    throw Exception("Failed to create texture object", err,
                             __FILE__, __FUNCTION__, __LINE__);
  }

  IMP_CUDA_CHECK();
}


} // namespace cu
} // namespace imp

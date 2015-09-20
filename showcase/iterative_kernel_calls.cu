#include "iterative_kernel_calls.cuh"
#include <cuda_runtime.h>

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
static inline void checkCudaErrorState(const char* file, const char* function,
                                       const int line)
{
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if( err != ::cudaSuccess )
  {
    std::cerr << "cuda error: " << cudaGetErrorString(err)
              << " [" << file << ", " << function << ", " << line << "]" << std::endl;
  }
}

//------------------------------------------------------------------------------
#ifdef IMP_THROW_ON_CUDA_ERROR
#  define CU_CHECK_ERROR() checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__)
#else
#  define CU_CHECK_ERROR() cudaDeviceSynchronize()
#endif

//#############################################################################
// texture fetch wrapper
template<typename T>
__device__ __forceinline__
T tex2DFetch(
    const Texture2D& tex, float x, float y,
    float mul_x=1.f, float mul_y=1.f, float add_x=0.f, float add_y=0.f)
{
  return ::tex2D<T>(tex.tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
}


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

//-----------------------------------------------------------------------------
__global__ void kernelCall(float* out_buffer, size_t stride,
                           cudaTextureObject_t in_tex,
                           size_t width, size_t height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x<width && y<height)
  {
    float val = ::tex2D<float>(in_tex, x+.5f, y+.5f);
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
  CU_CHECK_ERROR();

  dim3 dim_block = dim3(32,32);
  dim3 dim_grid(divUp(width_, dim_block.x), divUp(height_, dim_block.y));
  kernelCall
      <<< dim_grid, dim_block
      >>> (out_buffer_, pitch_/sizeof(float), *in_tex_, width_, height_);
  CU_CHECK_ERROR();


  cudaMallocPitch((void**)&unrelated_, &unrelated_pitch_, width_*sizeof(float), height_);
  CU_CHECK_ERROR();
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

  dim3 dim_block = dim3(32,32);
  dim3 dim_grid(divUp(width_, dim_block.x), divUp(height_, dim_block.y));

  // if this is called textured are messed up
  if (break_things)
  {
    this->breakThings();
  }

  kernelCall
      <<< dim_grid, dim_block
      >>> (out_buffer_, pitch_/sizeof(float), *in_tex_, width_, height_);
  CU_CHECK_ERROR();
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
    std::cerr << "Failed to create texture object: " << cudaGetErrorString(err)
              << " [" << __FILE__ << ", " << __FUNCTION__ << ", " << __LINE__ << "]" << std::endl;
  }

  CU_CHECK_ERROR();
}


} // namespace cu

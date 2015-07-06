#include <imp/cu_imgproc/iterative_kernel_calls.cuh>

#include <iostream>

#include <cuda_runtime.h>

#include <imp/core/pixel.hpp>
#include <imp/cu_core/cu_texture.cuh>
#include <imp/cu_core/cu_k_derivative.cuh>
#include <imp/cu_core/cu_math.cuh>
#include <imp/cu_core/cu_texture.cuh>

inline __host__ __device__ float2 operator*(float b, float2 a)
{
  return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ float2 operator/(float2 a, float b)
{
  return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ float dot(float2 a, float2 b)
{
  return a.x * b.x + a.y * b.y;
}


namespace imp {
namespace cu {

//-----------------------------------------------------------------------------
__global__ void k_ikcInit(Pixel32fC1* d_u, Pixel32fC1* d_u_prev, size_t stride_u,
                          Pixel32fC2* d_p, size_t stride_p,
                          imp::cu::Texture2D f_tex,
                          size_t width, size_t height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x<width && y<height)
  {
    float val = tex2DFetch<float>(f_tex, x, y);
    d_u[y*stride_u + x] = val;
    d_u_prev[y*stride_u + x] = val;
    d_p[y*stride_p + x] = Pixel32fC2(0.0f, 0.0f);
  }
}

//-----------------------------------------------------------------------------
__global__ void k_ikcOne(
    Pixel32fC1* d_u, Pixel32fC1* d_u_prev, size_t stride_u,
    Texture2D f_tex, Texture2D u_tex, Texture2D p_tex,
    float lambda, float tau, float theta, size_t width, size_t height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x<width && y<height)
  {
    float f = tex2DFetch<float>(f_tex, x, y);
    float u = tex2DFetch<float>(u_tex, x, y);
    float u_prev = u;
    float div = dpAd(p_tex, x, y, width, height);

    u = (u + tau*(div + lambda*f)) / (1.0f + tau*lambda);

    d_u[y*stride_u + x] = u;
    d_u_prev[y*stride_u + x] = u + theta*(u-u_prev);
  }
}

//-----------------------------------------------------------------------------
__global__ void k_ikcTwo(
    Pixel32fC2* d_p, size_t stride_p, Texture2D p_tex, Texture2D u_prev_tex,
    float sigma, size_t width, size_t height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x<width && y<height)
  {
    float2 p = tex2DFetch<float2>(p_tex, x, y);
    float2 dp_u = dp(u_prev_tex, x, y);

    p = p + sigma*dp_u;
    p = p / max(1.0f, length(p));
    d_p[y*stride_p + x] = {p.x, p.y};
  }
}

//#############################################################################

//-----------------------------------------------------------------------------
IterativeKernelCalls::IterativeKernelCalls()
  : f_tex_(nullptr)
  , u_tex_(nullptr)
  , u_prev_tex_(nullptr)
  , p_tex_(nullptr)
{
}

//-----------------------------------------------------------------------------
IterativeKernelCalls::~IterativeKernelCalls()
{

}

//-----------------------------------------------------------------------------
void IterativeKernelCalls::init(const Size2u& size)
{
  size_ = size;
  fragmentation_.reset(new Fragmentation(size));

  // setup internal memory
  this->u_.reset(new ImageGpu32fC1(size));
  this->u_prev_.reset(new ImageGpu32fC1(size));
  this->p_.reset(new ImageGpu32fC2(size));

  IMP_CUDA_CHECK();

  // setup textures
  f_tex_ = f_->genTexture(false, cudaFilterModeLinear, cudaAddressModeClamp,
                          (f_->bitDepth()==8) ? cudaReadModeNormalizedFloat :
                                                cudaReadModeElementType);
  u_tex_ = u_->genTexture(false, cudaFilterModeLinear, cudaAddressModeClamp,
                          cudaReadModeElementType);
  u_prev_tex_ = u_prev_->genTexture(false, cudaFilterModeLinear,
                                    cudaAddressModeClamp, cudaReadModeElementType);
  p_tex_ = p_->genTexture(false, cudaFilterModeLinear, cudaAddressModeClamp,
                          cudaReadModeElementType);
  IMP_CUDA_CHECK();

  // init internal vars
  k_ikcInit
      <<< dimGrid(), dimBlock() >>> (u_->data(), u_prev_->data(), u_->stride(),
                                     p_->data(), p_->stride(),
                                     *f_tex_, size_.width(), size_.height());


  unrelated_.reset(new ImageGpu32fC1(size_));
  unrelated_->setValue(0);


  IMP_CUDA_CHECK();
}

//-----------------------------------------------------------------------------
void IterativeKernelCalls::denoise(const std::shared_ptr<ImageBase>& dst,
                                   const std::shared_ptr<ImageBase>& src,
                                   bool break_things)
{
  if (params_.verbose)
  {
    std::cout << "[Solver @gpu] IterativeKernelCalls::denoise:" << std::endl;
  }

  if (src->size() != dst->size())
  {
    throw imp::cu::Exception("Input and output image are not of the same size.",
                             __FILE__, __FUNCTION__, __LINE__);
  }

  f_ = std::dynamic_pointer_cast<ImageGpu>(src);
  //! @todo (MWE) we could use dst for u_ if pixel_type is consistent

  if (size_ != f_->size())
  {
    this->init(f_->size());
  }

  // internal params
  float L = sqrtf(8.0f);
  float tau = 1/L;
  float sigma = 1/L;
  float theta = 1.0f;

  for(int iter = 0; iter < this->params_.max_iter; ++iter)
  {
    if (sigma < 1000.0f)
      theta = 1.f/sqrtf(1.0f+0.7f*this->params_.lambda*tau);
    else
      theta = 1.0f;

    if (params_.verbose)
    {
      std::cout << "(ikc solver) iter: " << iter << "; tau: " << tau
                << "; sigma: " << sigma << "; theta: " << theta << std::endl;
    }

    if (break_things)
    {
      this->breakThings();
    }

    k_ikcTwo
        <<< dimGrid(), dimBlock() >>> (p_->data(), p_->stride(),
                                       *p_tex_, *u_prev_tex_,
                                       sigma, size_.width(), size_.height());
    cudaThreadSynchronize();

    k_ikcOne
        <<< dimGrid(), dimBlock() >>> (u_->data(), u_prev_->data(), u_->stride(),
                                       *f_tex_, *u_tex_, *p_tex_,
                                       params_.lambda, tau, theta,
                                       size_.width(), size_.height());
    cudaThreadSynchronize();


    sigma /= theta;
    tau *= theta;

  }
  IMP_CUDA_CHECK();

  std::shared_ptr<ImageGpu32fC1> u(std::dynamic_pointer_cast<ImageGpu32fC1>(dst));
  u_->copyTo(*u);

  IMP_CUDA_CHECK();
}

//-----------------------------------------------------------------------------
void IterativeKernelCalls::breakThings()
{
  Pixel32fC1 val_min, val_max;
  imp::cu::minMax(*unrelated_, val_min, val_max);

  IMP_CUDA_CHECK();
}


} // namespace cu
} // namespace imp

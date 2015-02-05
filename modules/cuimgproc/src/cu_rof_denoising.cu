#include <imp/cuimgproc/cu_rof_denoising.cuh>

#include <iostream>

#include <cuda_runtime.h>

#include <imp/cuda_toolkit/helper_math.h>
#include <imp/core/pixel.hpp>
#include <imp/cucore/cu_texture.cuh>


namespace imp { namespace cu {

//-----------------------------------------------------------------------------
/** compute forward differences in x- and y- direction */
static __device__ __forceinline__ float2 dp(
    const imp::cu::Texture2D& tex, float x, float y, size_t width, size_t height)
{
  x+=0.5f;
  y+=0.5f;
  float2 grad = make_float2(0.0f, 0.0f);
  float cval = tex2D<float>(tex, x, y);
  if (x<width-1)
  {
    grad.x = tex2D<float>(tex, x+1.f, y) - cval;
  }
  if (y<height-1)
  {
    grad.y = tex2D<float>(tex, x, y+1.f) - cval;
  }
  return grad;
}

//-----------------------------------------------------------------------------
/** compute divergence using backward differences (adjugate from dp). */
static __device__ __forceinline__
float dpAd(const imp::cu::Texture2D& tex, size_t x, size_t y, size_t width, size_t height)
{
  float2 cval = tex2D<float2>(tex, x+0.5f, y+0.5f);
  float2 wval = tex2D<float2>(tex, x-0.5f, y+0.5f);
  float2 nval = tex2D<float2>(tex, x+0.5f, y-0.5f);

  if (x == 0)
    wval.x = 0.0f;
  else if (x >= width-1)
    cval.x = 0.0f;


  if (y == 0)
    nval.y = 0.0f;
  else if (y >= height-1)
    cval.y = 0.0f;

  return (cval.x - wval.x + cval.y - nval.y);
}

//// texture object is a kernel argument
//template<typename Pixel>
//__global__ void k_simpleTextureObjectTest(Pixel* u, size_t stride_u,
//                                          imp::cu::Texture2D f_tex,
//                                          Pixel* f, size_t stride_f,
//                                          size_t width, size_t height)
//{
//  int x = blockIdx.x*blockDim.x + threadIdx.x;
//  int y = blockIdx.y*blockDim.y + threadIdx.y;

//  if (x>=0 && y>=0 && x<width && y<height)
//  {
//    float px = tex2D<float>(f_tex, x+.5f, y+.5f);
//    u[y*stride_u+x] = f[y*stride_f+x] - static_cast<int>(255.0f*px);
//  }
//}

//-----------------------------------------------------------------------------
__global__ void k_solveRofPrimalIteration(
    float* d_u, float* d_u_prev, size_t stride_8uC1,
    Texture2D f_tex, Texture2D u_tex, Texture2D p_tex,
    float lambda, float tau, float theta, size_t width, size_t height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x<width && y<height)
  {
    float xx = x+0.5f;
    float yy = y+0.5f;

    float f = tex2D<float>(f_tex, xx, yy);
    float u = tex2D<float>(u_tex, xx, yy);
    float u_prev = u;
    float div = dpAd(p_tex, x, y, width, height);

    u = (u + tau*(div + lambda*f)) / (1.0f + tau*lambda);

    d_u[y*stride_8uC1 + x] = u;
    d_u_prev[y*stride_8uC1 + x] = u + theta*(u-u_prev);
  }
}

//-----------------------------------------------------------------------------
__global__ void k_solveRofDualIteration(
    float2* d_p, size_t stride_p, Texture2D p_tex, Texture2D u_prev_tex,
    float sigma, size_t width, size_t height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x<width && y<height)
  {
    float2 p = tex2D<float2>(p_tex, x+.5f, y+.5f);
    float2 dp_u = dp(u_prev_tex, x, y, width, height);

    p += sigma*dp_u;
    p /= max(1.0f, length(p));
    d_p[y*stride_p + x] = p;
  }
}

//-----------------------------------------------------------------------------
__global__ void k_convertResult8uC1(unsigned char* u, size_t stride_u,
                                    imp::cu::Texture2D u_tex,
                                    size_t width, size_t height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x<width && y<height)
  {
    u[y*stride_u + x] = static_cast<unsigned char>(
          255.0f * tex2D<float>(u_tex, x+.5f, y+.5f));
  }
}


//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
void RofDenoising<Pixel, pixel_type>::RofDenoising::denoise(ImagePtr f, ImagePtr u)
{
  std::cout << "solving the ROF image denosing model (gpu)" << std::endl;

  if (f->size() != u->size())
  {
    throw imp::cu::Exception("Input and output image are not of the same size.",
                             __FILE__, __FUNCTION__, __LINE__);
  }

  this->f_ = f;

  if (this->size_ != this->f_->size()
      || this->u_prev_ == nullptr
      || this->p_ == nullptr
      || fragmentation_ == nullptr)
  {
    this->size_ = this->f_->size();
    fragmentation_.reset(new Fragmentation<16>(this->size_));

    // setup internal memory
    switch (this->u_->nChannels())
    {
    case 1:
      this->u_.reset(new ImageGpu32fC1(this->size_));
      this->u_prev_.reset(new ImageGpu32fC1(this->size_));
      this->p_.reset(new ImageGpu32fC2(this->size_));
      break;
    default:
      throw imp::cu::Exception("ROF denoising not implemented for given image type.",
                               __FILE__, __FUNCTION__, __LINE__);
    }

    // setup textures
    this->f_tex_ = this->f_->genTexture(false,
                                        cudaFilterModeLinear,
                                        cudaAddressModeClamp,
                                        cudaReadModeNormalizedFloat);
    this->u_tex_ = this->u_->genTexture(false,
                                        cudaFilterModeLinear,
                                        cudaAddressModeClamp,
                                        cudaReadModeElementType);
    this->u_prev_tex_ = this->u_prev_->genTexture(false,
                                        cudaFilterModeLinear,
                                        cudaAddressModeClamp,
                                        cudaReadModeElementType);
    this->p_tex_ = this->p_->genTexture(false,
                                        cudaFilterModeLinear,
                                        cudaAddressModeClamp,
                                        cudaReadModeElementType);

    // internal params
    float L = sqrtf(8.0f);
    float tau = 1/L;
    float sigma = 1/L;
    float theta = 1.0f;

    for(int iter = 0; iter < this->params_.max_iter; ++iter)
    {
      k_solveRofDualIteration
          <<< fragmentation_->dimGrid, fragmentation_->dimBlock >>> (
          reinterpret_cast<float2*>(this->p_->data()), this->p_->stride(),
          *this->p_tex_, *this->u_prev_tex_,
          sigma, this->size_.width(), this->size_.height());

      if (sigma < 1000.0f)
        theta = 1.f/sqrtf(1.0f+0.7f*this->params_.lambda*tau);
      else
        theta = 1.0f;

      k_solveRofPrimalIteration
          <<< fragmentation_->dimGrid, fragmentation_->dimBlock >>> (
          reinterpret_cast<float*>(this->u_->data()),
          reinterpret_cast<float*>(this->u_prev_->data()), this->u_->stride(),
          *this->f_tex_, *this->u_tex_, *this->p_tex_,
          this->params_.lambda, tau, theta, this->size_.width(), this->size_.height());

      sigma /= theta;
      theta *= theta;
    }

    // copy final result to output
//    if (u->pixelType() == this->u_->pixelType())
//    {
//      this->u_->copyTo(*u);
//    }
//    else
//    {
      switch (u->pixelType())
      {
      case PixelType::i8uC1:
        k_convertResult8uC1
            <<< fragmentation_->dimGrid, fragmentation_->dimBlock >>> (
            reinterpret_cast<unsigned char*>(u->data()), u->stride(), *this->u_tex_,
            this->size_.width(), this->size_.height());
      }

//    }


//    // call test kernel
//    k_simpleTextureObjectTest <<< fragmentation_->dimGrid, fragmentation_->dimBlock >>> (
//      this->u_->data(), this->u_->stride(), *(this->f_tex_.get()),
//      this->f_->data(), this->f_->stride(),
//      this->size_.width(), this->size_.height());

  }
}

//=============================================================================
// Explicitely instantiate the desired classes
// (sync with typedefs at the end of the hpp file)
template class RofDenoising<imp::Pixel8uC1, imp::PixelType::i8uC1>;
template class RofDenoising<imp::Pixel32fC1, imp::PixelType::i32fC1>;

} // namespace cu
} // namespace imp

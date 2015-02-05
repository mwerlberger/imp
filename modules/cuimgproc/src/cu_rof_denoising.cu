#include <imp/cuimgproc/cu_rof_denoising.cuh>

#include <iostream>

#include <cuda_runtime.h>

#include <imp/core/pixel.hpp>


namespace imp { namespace cu {

// texture object is a kernel argument
template<typename Pixel>
__global__ void k_simpleTextureObjectTest(Pixel* u, size_t stride_u,
                                          imp::cu::Texture2D f_tex,
                                          Pixel* f, size_t stride_f,
                                          size_t width, size_t height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x>=0 && y>=0 && x<width && y<height)
  {
    float px = tex2D<float>(f_tex, x+.5f, y+.5f);
    u[y*stride_u+x] = f[y*stride_f+x] - static_cast<int>(255.0f*px);
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
  this->u_ = u;

  if (this->size_ != this->f_->size()
      || this->u_prev_ == nullptr
      || this->p_ == nullptr
      || fragmentation_ == nullptr)
  {
    this->size_ = this->f_->size();
    fragmentation_.reset(new Fragmentation<16>(this->size_));

    switch (this->u_->nChannels())
    {
    case 1:
      this->u_prev_.reset(new ImageGpu32fC1(this->size_));
      this->p_.reset(new ImageGpu32fC2(this->size_));
      break;
    default:
      throw imp::cu::Exception("ROF denoising not implemented for given image type.",
                               __FILE__, __FUNCTION__, __LINE__);
    }


    this->f_tex_ = this->f_->genTexture(false,
                                        cudaFilterModeLinear,
                                        cudaAddressModeClamp,
                                        cudaReadModeNormalizedFloat);

    // call test kernel
    k_simpleTextureObjectTest <<< fragmentation_->dimGrid, fragmentation_->dimBlock >>> (
      this->u_->data(), this->u_->stride(), *(this->f_tex_.get()),
      this->f_->data(), this->f_->stride(),
      this->size_.width(), this->size_.height());

  }
}

//=============================================================================
// Explicitely instantiate the desired classes
// (sync with typedefs at the end of the hpp file)
template class RofDenoising<imp::Pixel8uC1, imp::PixelType::i8uC1>;
template class RofDenoising<imp::Pixel32fC1, imp::PixelType::i32fC1>;

} // namespace cu
} // namespace imp

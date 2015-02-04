#include <imp/cuimgproc/cu_rof_denoising.cuh>

#include <iostream>
#include <cstring>

#include <cuda_runtime_api.h>

#include <imp/core/pixel.hpp>


namespace imp { namespace cu {

// texture object is a kernel argument
template<typename Pixel>
__global__ void k_simpleTextureObjectTest(Pixel* u, size_t stride_u,
                                          cudaTextureObject_t f_tex,
                                          Pixel* f, size_t stride_f,
                                          size_t width, size_t height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x>=0 && y>=0 && x<width && y<height)
  {
    float px = tex2D<float>(f_tex, x+.5f, y+0.5f);
    u[y*stride_u+x].x = f[y*stride_f+x].x - static_cast<int>(255.0f*px);
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


    // Use the texture object
    cudaResourceDesc f_tex_res;
    std::memset(&f_tex_res, 0, sizeof(f_tex_res));
    f_tex_res.resType = cudaResourceTypePitch2D;
    f_tex_res.res.pitch2D.devPtr = this->f_->data();
    f_tex_res.res.pitch2D.width = this->f_->width();
    f_tex_res.res.pitch2D.height = this->f_->height();
    f_tex_res.res.pitch2D.pitchInBytes = this->f_->pitch();
    f_tex_res.res.pitch2D.desc = cudaCreateChannelDesc<std::uint8_t>();

    cudaTextureDesc texDescr;
    std::memset(&texDescr, 0, sizeof(texDescr));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    //texDescr.addressMode[2] = addressMode;
    texDescr.readMode = cudaReadModeNormalizedFloat;

    cudaCreateTextureObject(&this->f_tex_, &f_tex_res, &texDescr, 0);

    // call test kernel
    k_simpleTextureObjectTest <<< fragmentation_->dimGrid, fragmentation_->dimBlock >>> (
      this->u_->data(), this->u_->stride(), this->f_tex_,
      this->f_->data(), this->f_->stride(),
      this->size_.width(), this->size_.height());



    // destroy texture objects
    cudaDestroyTextureObject(this->f_tex_);
  }
}

//=============================================================================
// Explicitely instantiate the desired classes
// (sync with typedefs at the end of the hpp file)
template class RofDenoising<imp::Pixel8uC1, imp::PixelType::i8uC1>;
template class RofDenoising<imp::Pixel32fC1, imp::PixelType::i32fC1>;

} // namespace cu
} // namespace imp

#include <imp/cuimgproc/cu_rof_denoising.cuh>
#include <iostream>

namespace imp { namespace cu {


//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
void RofDenoising<Pixel, pixel_type>::RofDenoising::denoise(ImagePtr f, ImagePtr u)
{
  std::cout << "solving the ROF image denosing model (gpu)" << std::endl;

  if (this->f_->size() != this->u_->size())
  {
    throw imp::cu::Exception("Input and output image are not of the same size.",
                             __FILE__, __FUNCTION__, __LINE__);
  }

  this->f_ = f;
  this->u_ = u;

  if (this->size_ != this->f_->size() || this->u_prev_ == nullptr)
  {
    this->size_ = this->f_->size();
    this->u_prev_.reset(new Image(this->size_));
  }
}

//=============================================================================
// Explicitely instantiate the desired classes
// (sync with typedefs at the end of the hpp file)
template class RofDenoising<imp::Pixel8uC1, imp::PixelType::i8uC1>;
template class RofDenoising<imp::Pixel32fC1, imp::PixelType::i32fC1>;

} // namespace cu
} // namespace imp

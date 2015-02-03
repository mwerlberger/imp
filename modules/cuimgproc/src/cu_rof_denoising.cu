#include <imp/cuimgproc/cu_rof_denoising.cuh>
#include <iostream>

namespace imp { namespace cu {


//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
void RofDenoising<Pixel, pixel_type>::RofDenoising::denoise(
    Base::ImagePtr f, Base::ImagePtr u)
{
  std::cout << "solving the ROF image denosing model (gpu)" << std::endl;

  if (f_->size() != u_->size())
  {
    throw imp::cu::Exception("Input and output image are not of the same size.",
                             __FILE__, __FUNCTION__, __LINE__);
  }

  f_ = f;
  u_ = u;

  if (size_ != f_->size() || u_prev_ == nullptr)
  {
    size_ = f_->size();
    u_prev_.reset(new Base::Image(size_));
  }
}

//=============================================================================
// Explicitely instantiate the desired classes
// (sync with typedefs at the end of the hpp file)
template class RofDenoising<imp::Pixel8uC1, imp::PixelType::i8uC1>;
template class RofDenoising<imp::Pixel32fC1, imp::PixelType::i32fC1>;

} // namespace cu
} // namespace imp

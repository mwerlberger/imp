#include <imp/cuimgproc/image_pyramid.hpp>

#include <imp/core/exception.hpp>
#include <imp/core/pixel_enums.hpp>
#include <imp/core/image.hpp>
#include <imp/core/image_raw.hpp>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cuimgproc/cu_image_transform.cuh>


namespace imp {


//------------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
ImagePyramid<Pixel,pixel_type>::ImagePyramid(ImagePtr img, float scale_factor,
                                             std::uint32_t size_bound,
                                             size_type max_num_levels)
  : scale_factor_(scale_factor)
  , size_bound_(size_bound)
  , max_num_levels_(max_num_levels)
{
  this->init(img->size());
  this->updateImage(img, imp::InterpolationMode::linear);
}

//------------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
void ImagePyramid<Pixel,pixel_type>::clear() noexcept
{
  levels_.clear();
  scale_factors_.clear();
}

//------------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
void ImagePyramid<Pixel,pixel_type>::init(const imp::Size2u& size)
{
  if (scale_factor_<=0.0 || scale_factor_>=1)
  {
    throw imp::Exception("Initializing image pyramid with scale factor <=0 or >=1 not possible.",
                         __FILE__, __FUNCTION__, __LINE__);
  }

  if (!levels_.empty() || scale_factors_.empty())
  {
    this->clear();
  }

  std::uint32_t shorter_side = std::min(size.width(), size.height());

  // calculate the maximum number of levels
  float ratio = static_cast<float>(shorter_side)/static_cast<float>(size_bound_);
  // +1 because the original size is level 0
  std::uint32_t possible_num_levels =
      static_cast<int>(-std::log(ratio)/std::log(scale_factor_)) + 1;
  num_levels_ = std::min(max_num_levels_, possible_num_levels);

  // init rate for each level
  for (size_type i = 0; i<num_levels_; ++i)
  {
    scale_factors_.push_back(std::pow(scale_factor_, static_cast<float>(i)));
  }

  std::cout << "img size: " << size.width() << "x" << size.height() << std::endl;
  std::cout << "num_levels: " << num_levels_ << std::endl;
  std::cout << " SCALE FACTORS:" << std::endl;
  for (float sf : scale_factors_)
  {
    std::cout << sf << std::endl;
  }
}

//------------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
void ImagePyramid<Pixel,pixel_type>::updateImage(ImagePtr img_level0,
                                                 InterpolationMode interp)
{
  // TODO sanity checks

  levels_.push_back(img_level0);
  Size2u sz0 =  img_level0->size();

  for (size_type i=1; i<num_levels_; ++i)
  {
    Size2u sz(static_cast<std::uint32_t>(sz0.width()*scale_factors_[i] + 0.5f),
              static_cast<std::uint32_t>(sz0.height()*scale_factors_[i] + 0.5f));

    // init level memory with either ImageGpu or ImageRaw
    if(img_level0->isGpuMemory())
    {
      ImageGpuPtr img = std::make_shared<imp::cu::ImageGpu<Pixel,pixel_type>>(sz);
      ImageGpuPtr prev = std::dynamic_pointer_cast<ImageGpu>(levels_.back());
      imp::cu::reduce(img.get(), prev.get(), interp, true);
      levels_.push_back(img);
    }
    else
    {
      ImageRawPtr img = std::make_shared<imp::ImageRaw<Pixel,pixel_type>>(sz);
      //! @todo (MWE) cpu reduction
      levels_.push_back(img);
    }



  }

}

//=============================================================================
// Explicitely instantiate the desired classes
// (sync with typedefs at the end of the hpp file)
template class ImagePyramid<imp::Pixel8uC1, imp::PixelType::i8uC1>;
template class ImagePyramid<imp::Pixel8uC2, imp::PixelType::i8uC2>;
//template class ImagePyramid<imp::Pixel8uC3, imp::PixelType::i8uC3>;
template class ImagePyramid<imp::Pixel8uC4, imp::PixelType::i8uC4>;

template class ImagePyramid<imp::Pixel16uC1, imp::PixelType::i16uC1>;
template class ImagePyramid<imp::Pixel16uC2, imp::PixelType::i16uC2>;
//template class ImagePyramid<imp::Pixel16uC3, imp::PixelType::i16uC3>;
template class ImagePyramid<imp::Pixel16uC4, imp::PixelType::i16uC4>;

template class ImagePyramid<imp::Pixel32sC1, imp::PixelType::i32sC1>;
template class ImagePyramid<imp::Pixel32sC2, imp::PixelType::i32sC2>;
//template class ImagePyramid<imp::Pixel32sC3, imp::PixelType::i32sC3>;
template class ImagePyramid<imp::Pixel32sC4, imp::PixelType::i32sC4>;

template class ImagePyramid<imp::Pixel32fC1, imp::PixelType::i32fC1>;
template class ImagePyramid<imp::Pixel32fC2, imp::PixelType::i32fC2>;
//template class ImagePyramid<imp::Pixel32fC3, imp::PixelType::i32fC3>;
template class ImagePyramid<imp::Pixel32fC4, imp::PixelType::i32fC4>;


} // namespace imp


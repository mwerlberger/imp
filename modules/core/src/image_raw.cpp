#include <imp/core/image_raw.hpp>

#include <iostream>

#include <imp/core/exception.hpp>


namespace imp {

//-----------------------------------------------------------------------------
template<typename PixelStorageType, imp::PixelType pixel_type>
ImageRaw<PixelStorageType, pixel_type>::ImageRaw(std::uint32_t width, std::uint32_t height)
  : Base(width, height)
{
  data_.reset(Memory::alignedAlloc(width, height, &pitch_));
}

//-----------------------------------------------------------------------------
template<typename PixelStorageType, imp::PixelType pixel_type>
ImageRaw<PixelStorageType, pixel_type>::ImageRaw(const imp::Size2u& size)
  : Base(size)
{
  data_.reset(Memory::alignedAlloc(size, &pitch_));
}

//-----------------------------------------------------------------------------
template<typename PixelStorageType, imp::PixelType pixel_type>
ImageRaw<PixelStorageType, pixel_type>::ImageRaw(const ImageRaw& from)
  : Base(from)
{
  data_.reset(Memory::alignedAlloc(this->width(), this->height(), &pitch_));
  from.copyTo(*this);
}

//-----------------------------------------------------------------------------
template<typename PixelStorageType, imp::PixelType pixel_type>
ImageRaw<PixelStorageType, pixel_type>::ImageRaw(const Image<PixelStorageType, pixel_type>& from)
  : Base(from)
{
  data_.reset(Memory::alignedAlloc(this->width(), this->height(), &pitch_));
  from.copyTo(*this);
}

//-----------------------------------------------------------------------------
template<typename PixelStorageType, imp::PixelType pixel_type>
ImageRaw<PixelStorageType, pixel_type>
::ImageRaw(pixel_container_t data, std::uint32_t width, std::uint32_t height,
           size_type pitch, bool use_ext_data_pointer)
  : Base(width, height)
{
  if (data == nullptr)
  {
    throw imp::Exception("input data not valid", __FILE__, __FUNCTION__, __LINE__);
  }

  if(use_ext_data_pointer)
  {
    // This uses the external data pointer as internal data pointer.
    auto dealloc_nop = [](pixel_container_t) { ; };
    data_ = std::unique_ptr<pixel_storage_t, Deallocator>(
          data, Deallocator(dealloc_nop));
    pitch_ = pitch;
  }
  else
  {
    data_.reset(Memory::alignedAlloc(this->width(), this->height(), &pitch_));
    size_type stride = pitch / sizeof(pixel_storage_t);

    if (this->bytes() == pitch*height)
    {
      std::copy(data, data+stride*height, data_.get());
    }
    else
    {
      for (std::uint32_t y=0; y<height; ++y)
      {
        for (std::uint32_t x=0; x<width; ++x)
        {
          data_.get()[y*this->stride()+x] = data[y*stride + x];
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------
template<typename PixelStorageType, imp::PixelType pixel_type>
PixelStorageType* ImageRaw<PixelStorageType, pixel_type>::data(
    std::uint32_t ox, std::uint32_t oy)
{
  if (ox > this->width() || oy > this->height())
  {
    throw imp::Exception("Request starting offset is outside of the image.", __FILE__, __FUNCTION__, __LINE__);
  }

  return &data_.get()[oy*this->stride() + ox];
}

//-----------------------------------------------------------------------------
template<typename PixelStorageType, imp::PixelType pixel_type>
const PixelStorageType* ImageRaw<PixelStorageType, pixel_type>::data(
    std::uint32_t ox, std::uint32_t oy) const
{
  if (ox > this->width() || oy > this->height())
  {
    throw imp::Exception("Request starting offset is outside of the image.", __FILE__, __FUNCTION__, __LINE__);
  }

  return reinterpret_cast<const pixel_container_t>(&data_.get()[oy*this->stride() + ox]);
}

//=============================================================================
// Explicitely instantiate the desired classes
// (sync with typedefs at the end of the hpp file)
template class ImageRaw<imp::Pixel8uC1, imp::PixelType::i8uC1>;
template class ImageRaw<imp::Pixel8uC2, imp::PixelType::i8uC2>;
template class ImageRaw<imp::Pixel8uC3, imp::PixelType::i8uC3>;
template class ImageRaw<imp::Pixel8uC4, imp::PixelType::i8uC4>;

template class ImageRaw<imp::Pixel16uC1, imp::PixelType::i16uC1>;
template class ImageRaw<imp::Pixel16uC2, imp::PixelType::i16uC2>;
template class ImageRaw<imp::Pixel16uC3, imp::PixelType::i16uC3>;
template class ImageRaw<imp::Pixel16uC4, imp::PixelType::i16uC4>;

template class ImageRaw<imp::Pixel32sC1, imp::PixelType::i32sC1>;
template class ImageRaw<imp::Pixel32sC2, imp::PixelType::i32sC2>;
template class ImageRaw<imp::Pixel32sC3, imp::PixelType::i32sC3>;
template class ImageRaw<imp::Pixel32sC4, imp::PixelType::i32sC4>;

template class ImageRaw<imp::Pixel32fC1, imp::PixelType::i32fC1>;
template class ImageRaw<imp::Pixel32fC2, imp::PixelType::i32fC2>;
template class ImageRaw<imp::Pixel32fC3, imp::PixelType::i32fC3>;
template class ImageRaw<imp::Pixel32fC4, imp::PixelType::i32fC4>;

} // namespace imp

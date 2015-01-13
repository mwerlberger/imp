#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <memory>
#include <algorithm>

#include <imp/core/image.hpp>
#include <imp/core/image_allocator.hpp>

namespace imp {

template<typename PixelStorageType, imp::PixelType pixel_type>
class ImageRaw : public imp::Image<PixelStorageType, pixel_type>
{
public:
  typedef Image<PixelStorageType, pixel_type> Base;
  typedef ImageRaw<PixelStorageType, pixel_type> ImRaw;
  typedef imp::ImageMemoryStorage<PixelStorageType> Memory;
  typedef imp::ImageMemoryDeallocator<PixelStorageType> Deallocator;

  typedef PixelStorageType pixel_storage_t;
  typedef pixel_storage_t* pixel_container_t;

  ImageRaw()
    : Base(pixel_type)
  { ; }

  virtual ~ImageRaw() = default;

  ImageRaw(std::uint32_t width, std::uint32_t height)
    : Base(width, height)
  {
    data_.reset(Memory::alignedAlloc(width, height, &pitch_));
  }

  ImageRaw(const imp::Size2u& size)
    : Base(size)
  {
    data_.reset(Memory::alignedAlloc(size, &pitch_));
  }

  ImageRaw(const ImageRaw& from)
    : Base(from)
  {
    data_.reset(Memory::alignedAlloc(this->width(), this->height(), &pitch_));
    if (this->bytes() == from.bytes())
    {
      std::cout << "using std::copy" << std::cout;
      std::copy(from.data_.get(), from.data_.get()+from.stride()*from.height(), data_.get());
    }
    else
    {
      std::cout << "pixel-wise copy" << std::cout;
      for (std::uint32_t y=0; y<this->height(); ++y)
      {
        for (std::uint32_t x=0; x<this->width(); ++x)
        {
          data_.get()[y*this->stride()+x] = from[y][x];
        }
      }
    }
  }

  ImageRaw(pixel_container_t data, std::uint32_t width, std::uint32_t height,
           size_type pitch, bool use_ext_data_pointer = false)
    : Base(width, height)
  {
    if (data == nullptr)
    {
      throw imp::Exception("input data not valid", __FILE__, __FUNCTION__, __LINE__);
    }

    if(use_ext_data_pointer)
    {
      // This uses the external data pointer as internal data pointer.
      auto dealloc_nop = [](pixel_container_t p) { ; };
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

  /** Returns a pointer to the pixel data.
   * The pointer can be offset to position \a (ox/oy).
   * @param[in] ox Horizontal offset of the pointer array.
   * @param[in] oy Vertical offset of the pointer array.
   * @return Pointer to the pixel array.
   */
  virtual pixel_container_t data(std::uint32_t ox = 0, std::uint32_t oy = 0) override
  {
    if (ox > this->width() || oy > this->height())
    {
      throw imp::Exception("Request starting offset is outside of the image.", __FILE__, __FUNCTION__, __LINE__);
    }

    return &data_.get()[oy*this->stride() + ox];
  }
  virtual const pixel_container_t data(std::uint32_t ox = 0, std::uint32_t oy = 0) const override
  {
    return reinterpret_cast<const pixel_container_t>(this->data(ox,oy));
  }

  void copyTo(Base& dst)
  {
    if (this->width() != dst.width() || this->height() != dst.height())
    {
      //! @todo (MWE) if width/height is the same but alignment is different we can copy manually!
      throw imp::Exception("Image size and/or memory alignment is different.", __FILE__, __FUNCTION__, __LINE__);
    }

    if (this->bytes() == dst.bytes())
    {
      std::cout << "using std::copy" << std::endl;
      std::copy(data_.get(), data_.get()+this->stride()*this->height(), dst.data());
    }
    else
    {
      std::cout << "pixel-wise copy" << std::endl;
      for (std::uint32_t y=0; y<this->height(); ++y)
      {
        for (std::uint32_t x=0; x<this->width(); ++x)
        {
          dst[y][x] = data_.get()[y*this->stride() + x];
        }
      }
    }
  }

  // :TODO:
  //ImageCpu& operator= (const ImageCpu<pixel_storage_type_t, Allocator>& from);

  /** Returns the (guaranteed) total amount of bytes saved in the data buffer. */
  virtual size_type bytes() const override
  {
    return this->height()*pitch_;
  }

  /** Returns the distance in bytes between starts of consecutive rows. */
  virtual size_type pitch() const override
  {
    return pitch_;
  }

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool isGpuMemory() const override
  {
    return false;
  }

protected:

  std::unique_ptr<pixel_storage_t, Deallocator > data_; //!< the actual image data
  size_type pitch_ = 0; //!< Row alignment in bytes.
};

typedef ImageRaw<std::uint8_t, imp::PixelType::i8uC1> ImageRaw8uC1;
typedef ImageRaw<std::uint16_t, imp::PixelType::i8uC1> ImageRaw16uC1;
typedef ImageRaw<std::int32_t, imp::PixelType::i8uC1> ImageRaw32sC1;
typedef ImageRaw<float, imp::PixelType::i8uC1> ImageRaw32fC1;

} // namespace imp


#endif // IMAGE_HPP

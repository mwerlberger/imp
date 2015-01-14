#ifndef IMP_IMAGE_HPP
#define IMP_IMAGE_HPP

#include <cstdint>

#include <imp/core/image_base.hpp>
#include <imp/core/exception.hpp>

namespace imp {

template<typename PixelStorageType, imp::PixelType pixel_type>
class Image : public ImageBase
{
  typedef PixelStorageType pixel_storage_t;
  typedef pixel_storage_t* pixel_container_t;

protected:
  Image(imp::PixelOrder pixel_order = imp::PixelOrder::undefined)
    : ImageBase(pixel_type, pixel_order)
  { ; }

  Image(std::uint32_t width, std::uint32_t height,
        PixelOrder pixel_order = imp::PixelOrder::undefined)
    : ImageBase(width, height, pixel_type, pixel_order)
  { ; }

  Image(const imp::Size2u &size,
        imp::PixelOrder pixel_order = imp::PixelOrder::undefined)
    : ImageBase(size, pixel_type, pixel_order)
  { ; }

  Image(const Image& from) = default;

public:
  virtual ~Image() = default;

  /** Returns a pointer to the pixel data.
   * The pointer can be offset to position \a (ox/oy).
   * @param[in] ox Horizontal offset of the pointer array.
   * @param[in] oy Vertical offset of the pointer array.
   * @return Pointer to the pixel array.
   */
  virtual pixel_container_t data(std::uint32_t ox = 0, std::uint32_t oy = 0) = 0;
  virtual const PixelStorageType* data(std::uint32_t ox = 0, std::uint32_t oy = 0) const = 0;

  /** Get Pixel value at position x,y. */
  pixel_storage_t pixel(std::uint32_t x, std::uint32_t y) const
  {
    return *data(x, y);
  }

  /** Get Pointer to beginning of row \a row (y index).
   * This enables the usage of [y][x] operator.
   */
  pixel_container_t operator[] (std::uint32_t row)
  {
    return data(0,row);
  }
  const pixel_container_t operator[] (std::uint32_t row) const
  {
    return data(0,row);
  }

  /**
   * @brief copyTo copies the internal image data to another class instance
   * @param dst Image class that will receive this image's data.
   */
  virtual void copyTo(Image& dst) const
  {
    if (this->width() != dst.width() || this->height() != dst.height())
    {
      throw imp::Exception("Copying failed: Image size differs.", __FILE__, __FUNCTION__, __LINE__);
    }

    if (this->bytes() == dst.bytes())
    {
      std::copy(this->data(), this->data()+this->stride()*this->height(), dst.data());
    }
    else
    {
      for (std::uint32_t y=0; y<this->height(); ++y)
      {
        for (std::uint32_t x=0; x<this->width(); ++x)
        {
          dst[y][x] = this->pixel(x,y);
        }
      }
    }
  }


  /** Returns the distnace in pixels between starts of consecutive rows. */
  virtual size_type stride() const override
  {
    return this->pitch()/sizeof(pixel_storage_t);
  }

  /** Returns the bit depth of the data pointer. */
  virtual std::uint8_t bitDepth() const override
  {
    return 8*sizeof(pixel_storage_t);
  }
};

//-----------------------------------------------------------------------------
// convenience typedefs
typedef Image<std::uint8_t, imp::PixelType::i8uC1> Image8uC1;
typedef Image<float, imp::PixelType::i32fC1> Image32fC1;

} // namespace imp


#endif // IMP_IMAGE_HPP

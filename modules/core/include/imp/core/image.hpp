#ifndef IMP_IMAGE_HPP
#define IMP_IMAGE_HPP

#include <cstdint>

#include <imp/core/image_base.hpp>
#include <imp/core/exception.hpp>
#include <imp/core/pixel.hpp>

namespace imp {

template<typename Pixel, imp::PixelType pixel_type>
class Image : public ImageBase
{
  typedef Pixel pixel_t;
  typedef pixel_t* pixel_container_t;

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
  Image() = delete;
  virtual ~Image() = default;

  /** Returns a pointer to the pixel data.
   * The pointer can be offset to position \a (ox/oy).
   * @param[in] ox Horizontal offset of the pointer array.
   * @param[in] oy Vertical offset of the pointer array.
   * @return Pointer to the pixel array.
   */
  virtual pixel_container_t data(std::uint32_t ox = 0, std::uint32_t oy = 0) = 0;
  virtual const Pixel* data(std::uint32_t ox = 0, std::uint32_t oy = 0) const = 0;

  /** Get Pixel value at position x,y. */
  pixel_t pixel(std::uint32_t x, std::uint32_t y) const
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
//  const pixel_container_t operator[] (std::uint32_t row) const
//  {
//    return data(0,row);
//  }

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

  /**
   * @brief copyFrom copies the image data from another class instance to this image
   * @param from Image class providing the image data.
   */
  virtual void copyFrom(const Image& from)
  {
    if (this->size()!= from.size())
    {
      throw imp::Exception("Copying failed: Image sizes differ.", __FILE__, __FUNCTION__, __LINE__);
    }

    if (this->bytes() == from.bytes())
    {
      std::copy(from.data(), from.data()+from.stride()*from.height(), this->data());
    }
    else
    {
      for (std::uint32_t y=0; y<this->height(); ++y)
      {
        for (std::uint32_t x=0; x<this->width(); ++x)
        {
          (*this)[y][x] = from.pixel(y,x);
        }
      }
    }
  }

  /** Returns the length of a row (not including the padding!) in bytes. */
  virtual size_type rowBytes() const
  {
    return this->width() * sizeof(pixel_t);
  }

  /** Returns the distnace in pixels between starts of consecutive rows. */
  virtual size_type stride() const override
  {
    return this->pitch()/sizeof(pixel_t);
  }

  /** Returns the bit depth of the data pointer. */
  virtual std::uint8_t bitDepth() const override
  {
    return 8*sizeof(pixel_t);
  }
};

//-----------------------------------------------------------------------------
// convenience typedefs
typedef Image<imp::Pixel8uC1, imp::PixelType::i8uC1> Image8uC1;
typedef Image<imp::Pixel8uC2, imp::PixelType::i8uC2> Image8uC2;
typedef Image<imp::Pixel8uC3, imp::PixelType::i8uC3> Image8uC3;
typedef Image<imp::Pixel8uC4, imp::PixelType::i8uC4> Image8uC4;

typedef Image<imp::Pixel16uC1, imp::PixelType::i16uC1> Image16uC1;
typedef Image<imp::Pixel16uC2, imp::PixelType::i16uC2> Image16uC2;
typedef Image<imp::Pixel16uC3, imp::PixelType::i16uC3> Image16uC3;
typedef Image<imp::Pixel16uC4, imp::PixelType::i16uC4> Image16uC4;

typedef Image<imp::Pixel32sC1, imp::PixelType::i32sC1> Image32sC1;
typedef Image<imp::Pixel32sC2, imp::PixelType::i32sC2> Image32sC2;
typedef Image<imp::Pixel32sC3, imp::PixelType::i32sC3> Image32sC3;
typedef Image<imp::Pixel32sC4, imp::PixelType::i32sC4> Image32sC4;

typedef Image<imp::Pixel32fC1, imp::PixelType::i32fC1> Image32fC1;
typedef Image<imp::Pixel32fC2, imp::PixelType::i32fC2> Image32fC2;
typedef Image<imp::Pixel32fC3, imp::PixelType::i32fC3> Image32fC3;
typedef Image<imp::Pixel32fC4, imp::PixelType::i32fC4> Image32fC4;

} // namespace imp


#endif // IMP_IMAGE_HPP

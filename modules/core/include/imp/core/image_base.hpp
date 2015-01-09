#ifndef IMP_IMAGE_BASE_HPP
#define IMP_IMAGE_BASE_HPP

//#include "globaldefs.h"
//#include "coredefs.h"

#include <imp/core/pixel_enums.hpp>
#include <imp/core/size.hpp>

namespace imp {

//! \todo We maybe do not want to have the Image class in the public dll interface??
class ImageBase
{
protected:
  ImageBase(PixelType pixel_type) :
    pixel_type_(pixel_type), size_(0,0), roi_(0,0,0,0)
  {
  }

  ImageBase(const ImageBase &from) :
    pixel_type_(from.pixelType()), size_(from.size_), roi_(from.roi_)
  {
  }

  ImageBase(PixelType pixel_type, unsigned int width, unsigned int height) :
      pixel_type_(pixel_type), size_(width, height), roi_(0, 0, width, height)
  {
  }

  ImageBase(PixelType pixel_type, const Size &size) :
      pixel_type_(pixel_type), size_(size), roi_(0, 0, size.width, size.height)
  {
  }

public:

  virtual ~ImageBase()
  {
  }

  ImageBase& operator= (const ImageBase &from)
  {
    // TODO == operator
    this->pixel_type_ = from.pixel_type_;
    this->size_ = from.size_;
    this->roi_ = from.roi_;
    return *this;
  }

  void setRoi(const Rect& roi)
  {
    roi_ = roi;
  }

  /** Returns the element types. */
  PixelType pixelType() const
  {
    return pixel_type_;
  }

  Size size() const
  {
    return size_;
  }

  Rect roi() const
  {
    return roi_;
  }

  unsigned int width() const
  {
    return size_.width;
  }

  unsigned int height() const
  {
    return size_.height;
  }

  /** Returns the number of pixels in the image. */
  size_t numel() const
  {
    return (size_.width * size_.height);
  }

  /** Returns the total amount of bytes saved in the data buffer. */
  virtual size_t bytes() const = 0;

  /** Returns the distance in bytes between starts of consecutive rows. */
  virtual std::uint32_t pitch() const = 0;

  /** Returns the distnace in pixels between starts of consecutive rows. */
  virtual std::uint32_t stride() const = 0;

  /** Returns the bit depth of the data pointer. */
  virtual std::uint8_t bitDepth() const = 0;

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool isGpuMemory() const = 0;

private:
  PixelType pixel_type_;
  PixelOrder pixel_order_;
  Size size_;
  Rect roi_;
};

} // namespace iuprivate

#endif // IMP_IMAGE_H

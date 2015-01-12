/*
 * Copyright (c) ICG. All rights reserved.
 *
 * Institute for Computer Graphics and Vision
 * Graz University of Technology / Austria
 *
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notices for more information.
 *
 *
 * Project     : ImageUtilities
 * Module      : Core
 * Class       : Image
 * Language    : C++
 * Description : Definition of image class for Ipp
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IMP_IMAGE_HPP
#define IMP_IMAGE_HPP

#include <imp/core/image_base.hpp>

namespace imp {

template<typename _PixelStorageType, imp::PixelType pixel_type>
class Image : public ImageBase
{
  typedef _PixelStorageType pixel_storage_t;
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
  pixel_container_t* data(std::uint32_t ox = 0, std::uint32_t oy = 0) = 0;
  const pixel_container_t* data(std::uint32_t ox = 0, std::uint32_t oy = 0) const = 0;

  /** Get Pixel value at position x,y. */
  pixel_storage_t pixel(std::uint32_t x, std::uint32_t y)
  {
    return *data(x, y);
  }

  /** Get Pointer to beginning of row \a row (y index).
   * This enables the usage of [y][x] operator.
   */
  pixel_container_t operator[] (std::uint32_t y)
  {
    return data_(0,y);
  }

  //! @todo (MWE)
  //Image& operator= (const Image<PixelType, Allocator>& from);

  /** Returns the distnace in pixels between starts of consecutive rows. */
  virtual size_t stride() const override
  {
    return pitch_/sizeof(pixel_storage_t);
  }

  /** Returns the bit depth of the data pointer. */
  virtual unsigned int bitDepth() const override
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

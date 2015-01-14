#ifndef IMP_PIXEL_ENUMS_HPP
#define IMP_PIXEL_ENUMS_HPP

namespace imp {

/**
 * @brief The PixelType enum defines a pixel's bit depth and number of channels.
 */
enum class PixelType
{
  undefined = -1,
  // interleaved pixel types (i prefix)
  i8uC1,
  i8uC2,
  i8uC3,
  i8uC4,
  i16uC1,
  i16uC2,
  i16uC3,
  i16uC4,
  i32uC1,
  i32uC2,
  i32uC4,
  i32sC1,
  i32sC2,
  i32sC3,
  i32sC4,
  i32fC1,
  i32fC2,
  i32fC3,
  i32fC4
};

/**
 * @brief The PixelOrder enum defines a pixel's channel ordering.
 */
enum class PixelOrder
{
  // interleaved (or single) channel pixel types
  undefined = -1,
  gray,
  rgb,
  bgr,
  rgba,
  bgra
};


} // namespace imp

#endif // IMP_PIXEL_ENUMS_HPP


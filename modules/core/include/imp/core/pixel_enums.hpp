#ifndef IMP_PIXEL_ENUMS_HPP
#define IMP_PIXEL_ENUMS_HPP

namespace imp {

/**
 * @brief The PixelType enum defines a pixel's bit depth and number of channels.
 */
enum class PixelType
{
  undefined = -1,
  8uC1,
  8uC2,
  8uC3,
  8uC4,
  16uC1,
  16uC2,
  16uC3,
  16uC4,
  32uC1,
  32uC2,
  32uC4,
  32sC1,
  32sC2,
  32sC3,
  32sC4,
  32fC1,
  32fC2,
  32fC3,
  32fC4
};

/**
 * @brief The PixelOrder enum defines a pixel's channel ordering.
 */
enum class PixelOrder
{
  // interleaved (or single) channel pixel types
  gray,
  rgb,
  bgr,
  rgba,
  bgra
};


} // namespace imp

#endif // IMP_PIXEL_ENUMS_HPP


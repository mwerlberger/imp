#ifndef IMAGE_RAW_HPP
#define IMAGE_RAW_HPP

#include <memory>
#include <algorithm>

#include <imp/core/image.hpp>
#include <imp/core/image_allocator.hpp>

namespace imp {

/**
 * @brief The ImageRaw class
 */
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

public:
  ImageRaw() = default;
  virtual ~ImageRaw() = default;

  ImageRaw(std::uint32_t width, std::uint32_t height);
  ImageRaw(const imp::Size2u& size);
  ImageRaw(const ImageRaw& from);
  ImageRaw(const Base& from);
  ImageRaw(pixel_container_t data, std::uint32_t width, std::uint32_t height,
           size_type pitch, bool use_ext_data_pointer = false);

  /** Returns a pointer to the pixel data.
   * The pointer can be offset to position \a (ox/oy).
   * @param[in] ox Horizontal offset of the pointer array.
   * @param[in] oy Vertical offset of the pointer array.
   * @return Pointer to the pixel array.
   */
  virtual PixelStorageType* data(std::uint32_t ox = 0, std::uint32_t oy = 0) override;
  virtual const PixelStorageType* data(std::uint32_t ox = 0, std::uint32_t oy = 0) const override;

  /** Returns the distance in bytes between starts of consecutive rows. */
  virtual size_type pitch() const override { return pitch_; }

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool isGpuMemory() const override { return false; }

protected:
  std::unique_ptr<pixel_storage_t, Deallocator > data_; //!< the actual image data
  size_type pitch_ = 0; //!< Row alignment in bytes.
};

//-----------------------------------------------------------------------------
// convenience typedefs
// (sync with explicit template class instantiations at the end of the cpp file)
typedef ImageRaw<imp::Pixel8uC1, imp::PixelType::i8uC1> ImageRaw8uC1;
typedef ImageRaw<imp::Pixel8uC2, imp::PixelType::i8uC2> ImageRaw8uC2;
typedef ImageRaw<imp::Pixel8uC3, imp::PixelType::i8uC3> ImageRaw8uC3;
typedef ImageRaw<imp::Pixel8uC4, imp::PixelType::i8uC4> ImageRaw8uC4;

typedef ImageRaw<imp::Pixel16uC1, imp::PixelType::i16uC1> ImageRaw16uC1;
typedef ImageRaw<imp::Pixel16uC2, imp::PixelType::i16uC2> ImageRaw16uC2;
typedef ImageRaw<imp::Pixel16uC3, imp::PixelType::i16uC3> ImageRaw16uC3;
typedef ImageRaw<imp::Pixel16uC4, imp::PixelType::i16uC4> ImageRaw16uC4;

typedef ImageRaw<imp::Pixel32sC1, imp::PixelType::i32sC1> ImageRaw32sC1;
typedef ImageRaw<imp::Pixel32sC2, imp::PixelType::i32sC2> ImageRaw32sC2;
typedef ImageRaw<imp::Pixel32sC3, imp::PixelType::i32sC3> ImageRaw32sC3;
typedef ImageRaw<imp::Pixel32sC4, imp::PixelType::i32sC4> ImageRaw32sC4;

typedef ImageRaw<imp::Pixel32fC1, imp::PixelType::i32fC1> ImageRaw32fC1;
typedef ImageRaw<imp::Pixel32fC2, imp::PixelType::i32fC2> ImageRaw32fC2;
typedef ImageRaw<imp::Pixel32fC3, imp::PixelType::i32fC3> ImageRaw32fC3;
typedef ImageRaw<imp::Pixel32fC4, imp::PixelType::i32fC4> ImageRaw32fC4;

} // namespace imp


#endif // IMAGE_RAW_HPP

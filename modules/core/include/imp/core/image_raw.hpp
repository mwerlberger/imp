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
  virtual pixel_container_t data(std::uint32_t ox = 0, std::uint32_t oy = 0) override;
  virtual const PixelStorageType* data(std::uint32_t ox = 0, std::uint32_t oy = 0) const override;

  /** Returns the distance in bytes between starts of consecutive rows. */
  virtual size_type pitch() const override { return pitch_; }

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool isGpuMemory() const override { return false; }

protected:
  std::unique_ptr<pixel_storage_t, Deallocator > data_; //!< the actual image data
  size_type pitch_ = 0; //!< Row alignment in bytes.
};

// convenience typedefs
// (sync with explicit template class instantiations at the end of the cpp file)
typedef ImageRaw<std::uint8_t, imp::PixelType::i8uC1> ImageRaw8uC1;
typedef ImageRaw<std::uint16_t, imp::PixelType::i8uC1> ImageRaw16uC1;
typedef ImageRaw<std::int32_t, imp::PixelType::i8uC1> ImageRaw32sC1;
typedef ImageRaw<float, imp::PixelType::i8uC1> ImageRaw32fC1;

} // namespace imp


#endif // IMAGE_RAW_HPP

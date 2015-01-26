#ifndef IMP_IMAGE_CV_HPP
#define IMP_IMAGE_CV_HPP

//#include <memory>
//#include <algorithm>

#include <opencv2/core/core.hpp>

#include <imp/core/image.hpp>
//#include <imp/core/image_allocator.hpp>
#include <imp/core/pixel_enums.hpp>

namespace imp {

/**
 * @brief The ImageCv class
 */
template<typename PixelStorageType, imp::PixelType pixel_type>
class ImageCv : public imp::Image<PixelStorageType, pixel_type>
{
public:
  typedef Image<PixelStorageType, pixel_type> Base;
  typedef ImageCv<PixelStorageType, pixel_type> ImCv;

  typedef PixelStorageType pixel_storage_t;
  typedef pixel_storage_t* pixel_container_t;

public:
  ImageCv() = default;
  virtual ~ImageCv() = default;

  ImageCv(std::uint32_t width, std::uint32_t height);
  ImageCv(const imp::Size2u& size);
  ImageCv(const ImCv& from);
  ImageCv(const Base& from);
  ImageCv(cv::Mat mat, imp::PixelOrder pixel_order_=imp::PixelOrder::undefined);
//  ImageCv(pixel_container_t data, std::uint32_t width, std::uint32_t height,
//          size_type pitch, bool use_ext_data_pointer = false);

  /** Returns the internal OpenCV image/mat
   */
  virtual cv::Mat cvMat();
  virtual const cv::Mat& cvMat() const;

  /** Returns a pointer to the pixel data.
   * The pointer can be offset to position \a (ox/oy).
   * @param[in] ox Horizontal offset of the pointer array.
   * @param[in] oy Vertical offset of the pointer array.
   * @return Pointer to the pixel array.
   */
  virtual pixel_container_t data(std::uint32_t ox = 0, std::uint32_t oy = 0) override;
  virtual const PixelStorageType* data(std::uint32_t ox = 0, std::uint32_t oy = 0) const override;

  /** Returns the distance in bytes between starts of consecutive rows. */
  virtual size_type pitch() const override { return m_mat.step; }

  /** Returns the bit depth of the opencv matrix elements. */
  virtual std::uint8_t bitDepth() const override {return 8*m_mat.elemSize(); }

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool isGpuMemory() const override { return false; }

protected:
  cv::Mat m_mat;
};

// convenience typedefs
// (sync with explicit template class instantiations at the end of the cpp file)
typedef ImageCv<std::uint8_t, imp::PixelType::i8uC1> ImageCv8uC1;
typedef ImageCv<std::uint16_t, imp::PixelType::i8uC1> ImageCv16uC1;
typedef ImageCv<std::int32_t, imp::PixelType::i8uC1> ImageCv32sC1;
typedef ImageCv<float, imp::PixelType::i8uC1> ImageCv32fC1;

} // namespace imp


#endif // IMP_IMAGE_CV_HPP

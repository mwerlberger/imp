#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <imp/core/image.hpp>
#include <imp/core/image_allocator.hpp>

namespace imp {

template<typename _PixelStorageType, imp::PixelType pixel_type>
class ImageRaw : public imp::Image<_PixelStorageType, pixel_type>
{
public:
  typedef _PixelStorageType pixel_storage_t;
  typedef pixel_storage_t* pixel_container_t;

  ImageRaw() :
    Image(_pixel_type),
    data_(0), pitch_(0), ext_data_pointer_(false)
  {
  }

  virtual ~ImageRaw()
  {
    if(!ext_data_pointer_)
    {
      // do not delete externally handeled data pointers.
      Allocator::free(data_);
      data_ = 0;
    }
    pitch_ = 0;
  }

  ImageRaw(unsigned int _width, unsigned int _height) :
    Image(_pixel_type, _width, _height), data_(0), pitch_(0),
    ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(_width, _height, &pitch_);
  }

  ImageRaw(const IuSize& size) :
    Image(_pixel_type, size.width, size.height), data_(0), pitch_(0),
    ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(size.width, size.height, &pitch_);
  }

  ImageRaw(const ImageRaw<pixel_storage_type_t, Allocator, _pixel_type>& from) :
    Image(from), data_(0), pitch_(0),
    ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(width(), height(), &pitch_);
    Allocator::copy(from.data(), from.pitch(), data_, pitch_, this->size());
  }

  ImageRaw(pixel_container_t _data, unsigned int _width, unsigned int _height,
           size_t _pitch, bool ext_data_pointer = false) :
    Image(_pixel_type, _width, _height), data_(0), pitch_(0),
    ext_data_pointer_(ext_data_pointer)
  {
    if(ext_data_pointer_)
    {
      // This uses the external data pointer as internal data pointer.
      data_ = _data;
      pitch_ = _pitch;
    }
    else
    {
      data_ = Allocator::alloc(width(), height(), &pitch_);
      Allocator::copy(_data, _pitch, data_, pitch_, this->size());
    }
  }

  /** Returns a pointer to the pixel data.
   * The pointer can be offset to position \a (ox/oy).
   * @param[in] ox Horizontal offset of the pointer array.
   * @param[in] oy Vertical offset of the pointer array.
   * @return Pointer to the pixel array.
   */
  pixel_container_t data(int ox = 0, int oy = 0)
  {
    return &data_[oy * stride() + ox];
  }
  const pixel_container_t data(int ox = 0, int oy = 0) const
  {
    return reinterpret_cast<const pixel_container_t>(
          &data_[oy * stride() + ox]);
  }

  /** Get Pixel value at position x,y. */
  pixel_storage_type_t getPixel(unsigned int x, unsigned int y)
  {
    return *data(x, y);
  }

  /** Get Pointer to beginning of row \a row (y index).
   * This enables the usage of [y][x] operator.
   */
  pixel_container_t operator[](unsigned int row)
  {
    return data_+row*stride();
  }

  // :TODO:
  //ImageCpu& operator= (const ImageCpu<pixel_storage_type_t, Allocator>& from);

  /** Returns the total amount of bytes saved in the data buffer. */
  virtual size_t bytes() const
  {
    return height()*pitch_;
  }

  /** Returns the distance in bytes between starts of consecutive rows. */
  virtual size_t pitch() const
  {
    return pitch_;
  }


  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool onDevice() const
  {
    return false;
  }

protected:
  std::unique_ptr<pixel_storage_t, > data_;
  pixel_container_t data_;
  size_t pitch_;
  bool ext_data_pointer_; /**< Flag if data pointer is handled outside the image class. */
};

} // namespace imp


#endif // IMAGE_HPP

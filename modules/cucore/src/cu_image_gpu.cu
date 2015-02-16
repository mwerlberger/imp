#include <imp/cucore/cu_image_gpu.cuh>

#include <iostream>
#include <memory>

#include <imp/cucore/cu_exception.hpp>
#include <imp/cucore/cu_utils.hpp>
#include <imp/cucore/cu_linearmemory.cuh>
#include <imp/cucore/cu_texture.cuh>
//#include <imp/cucore/cu_pixel_conversion.hpp>

// kernel includes
#include <imp/cucore/cu_k_setvalue.cuh>


namespace imp { namespace cu {

//template<typename Pixel, imp::PixelType pixel_type>
//const double ImageGpu<Pixel, pixel_type>::BLA = 3.18;

//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
ImageGpu<Pixel, pixel_type>::ImageGpu(std::uint32_t width, std::uint32_t height)
  : Base(width, height)
{
  this->initMemory();
}

//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
ImageGpu<Pixel, pixel_type>::ImageGpu(const imp::Size2u& size)
  : Base(size)
{
  this->initMemory();
}

//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
ImageGpu<Pixel, pixel_type>::ImageGpu(const ImageGpu& from)
  : Base(from)
{
  this->initMemory();
  this->copyFrom(from);
}

//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
ImageGpu<Pixel, pixel_type>::ImageGpu(const Image<Pixel, pixel_type>& from)
  : Base(from)
{
  this->initMemory();
  this->copyFrom(from);
}

////-----------------------------------------------------------------------------
//template<typename Pixel, imp::PixelType pixel_type>
//ImageGpu<Pixel, pixel_type>
//::ImageGpu(pixel_container_t data, std::uint32_t width, std::uint32_t height,
//           size_type pitch, bool use_ext_data_pointer)
//  : Base(width, height)
//{
//  if (data == nullptr)
//  {
//    throw imp::cu::Exception("input data not valid", __FILE__, __FUNCTION__, __LINE__);
//  }

//  if(use_ext_data_pointer)
//  {
//    // This uses the external data pointer as internal data pointer.
//    auto dealloc_nop = [](pixel_container_t) { ; };
//    data_ = std::unique_ptr<pixel_t, Deallocator>(
//          data, Deallocator(dealloc_nop));
//    pitch_ = pitch;
//  }
//  else
//  {
//    data_.reset(Memory::alignedAlloc(this->width(), this->height(), &pitch_));
//    size_type stride = pitch / sizeof(pixel_t);

//    if (this->bytes() == pitch*height)
//    {
//      std::copy(data, data+stride*height, data_.get());
//    }
//    else
//    {
//      for (std::uint32_t y=0; y<height; ++y)
//      {
//        for (std::uint32_t x=0; x<width; ++x)
//        {
//          data_.get()[y*this->stride()+x] = data[y*stride + x];
//        }
//      }
//    }
//  }
//}

//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
ImageGpu<Pixel, pixel_type>::~ImageGpu()
{
  //  delete gpu_data_;
}

//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
void ImageGpu<Pixel, pixel_type>::initMemory()
{
  data_.reset(Memory::alignedAlloc(this->size(), &pitch_));
  channel_format_desc_ = toCudaChannelFormatDesc(pixel_type);
}

//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
void ImageGpu<Pixel, pixel_type>::setRoi(const imp::Roi2u& roi)
{
  this->roi_ = roi;
  //  gpu_data_.roi = roi;
}

//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
void ImageGpu<Pixel, pixel_type>::copyTo(imp::Image<Pixel, pixel_type>& dst) const
{
  if (this->size()!= dst.size())
  {
    throw imp::Exception("Copying failed: Image sizes differ.", __FILE__, __FUNCTION__, __LINE__);
  }
  cudaMemcpyKind memcpy_kind = dst.isGpuMemory() ? cudaMemcpyDeviceToDevice :
                                                   cudaMemcpyDeviceToHost;
  const cudaError cu_err = cudaMemcpy2D(dst.data(), dst.pitch(),
                                        this->data(), this->pitch(),
                                        this->rowBytes(),
                                        this->height(), memcpy_kind);
  if (cu_err != cudaSuccess)
  {
    throw imp::cu::Exception("copyFrom failed", cu_err, __FILE__, __FUNCTION__, __LINE__);
  }

}

//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
void ImageGpu<Pixel, pixel_type>::copyFrom(const Image<Pixel, pixel_type>& from)
{
  if (this->size()!= from.size())
  {
    throw imp::Exception("Copying failed: Image sizes differ.", __FILE__, __FUNCTION__, __LINE__);
  }
  cudaMemcpyKind memcpy_kind = from.isGpuMemory() ? cudaMemcpyDeviceToDevice :
                                                    cudaMemcpyHostToDevice;
  const cudaError cu_err = cudaMemcpy2D(this->data(), this->pitch(),
                                        from.data(), from.pitch(),
                                        this->rowBytes(),
                                        this->height(), memcpy_kind);
  if (cu_err != cudaSuccess)
  {
    throw imp::cu::Exception("copyFrom failed", cu_err, __FILE__, __FUNCTION__, __LINE__);
  }
}

//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
Pixel* ImageGpu<Pixel, pixel_type>::data(
    std::uint32_t ox, std::uint32_t oy)
{
  //  if (ox > this->width() || oy > this->height())
  //  {
  //    throw imp::cu::Exception("Request starting offset is outside of the image.", __FILE__, __FUNCTION__, __LINE__);
  //  }

  if (ox != 0 || oy != 0)
  {
    throw imp::cu::Exception("Device memory pointer offset is not possible from host function");
  }

  return data_.get();
}

//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
const Pixel* ImageGpu<Pixel, pixel_type>::data(
    std::uint32_t ox, std::uint32_t oy) const
{
  if (ox != 0 || oy != 0)
  {
    throw imp::cu::Exception("Device memory pointer offset is not possible from host function");
  }

  //  return reinterpreft_cast<const pixel_container_t>(data_.get());
  return data_.get();
}

////-----------------------------------------------------------------------------
//template<typename Pixel, imp::PixelType pixel_type>
//void* ImageGpu<Pixel, pixel_type>::cuData()
//{
//  return (void*)data_.get();
//}

//-----------------------------------------------------------------------------
//template<typename Pixel, imp::PixelType pixel_type>
//const void* ImageGpu<Pixel, pixel_type>::cuData() const
//{
//  return (const void*)data_.get();
//}

template<typename Pixel, imp::PixelType pixel_type>
auto ImageGpu<Pixel, pixel_type>::cuData() -> decltype(imp::cu::toCudaVectorType(this->data()))
{
  return imp::cu::toCudaVectorType(this->data());
}

//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
void ImageGpu<Pixel, pixel_type>::setValue(const pixel_t& value)
{
  if (sizeof(pixel_t) == 1)
  {
    cudaMemset2D((void*)this->data(), this->pitch(), (int)value.c[0], this->rowBytes(), this->height());
  }
  else
  {
    // fragmentation
    cu::Fragmentation<16> frag(this->size());

    // todo add roi to kernel!
    imp::cu::k_setValue
        <<< frag.dimGrid, frag.dimBlock >>> (this->data(), this->stride(), value,
                                             this->width(), this->height());
  }
}

//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
std::unique_ptr<Texture2D> ImageGpu<Pixel, pixel_type>::genTexture(bool normalized_coords,
                                                                   cudaTextureFilterMode filter_mode,
                                                                   cudaTextureAddressMode address_mode,
                                                                   cudaTextureReadMode read_mode)
{
  return std::unique_ptr<Texture2D>(new Texture2D(this->cuData(), this->pitch(),
                                                  channel_format_desc_, this->size(),
                                                  normalized_coords, filter_mode,
                                                  address_mode, read_mode));
}


//=============================================================================
// Explicitely instantiate the desired classes
// (sync with typedefs at the end of the hpp file)
template class ImageGpu<imp::Pixel8uC1, imp::PixelType::i8uC1>;
template class ImageGpu<imp::Pixel8uC2, imp::PixelType::i8uC2>;
// be careful with 8uC3 images as pitch values are not divisable by 3!
template class ImageGpu<imp::Pixel8uC3, imp::PixelType::i8uC3>;
template class ImageGpu<imp::Pixel8uC4, imp::PixelType::i8uC4>;

template class ImageGpu<imp::Pixel16uC1, imp::PixelType::i16uC1>;
template class ImageGpu<imp::Pixel16uC2, imp::PixelType::i16uC2>;
template class ImageGpu<imp::Pixel16uC3, imp::PixelType::i16uC3>;
template class ImageGpu<imp::Pixel16uC4, imp::PixelType::i16uC4>;

template class ImageGpu<imp::Pixel32sC1, imp::PixelType::i32sC1>;
template class ImageGpu<imp::Pixel32sC2, imp::PixelType::i32sC2>;
template class ImageGpu<imp::Pixel32sC3, imp::PixelType::i32sC3>;
template class ImageGpu<imp::Pixel32sC4, imp::PixelType::i32sC4>;

template class ImageGpu<imp::Pixel32fC1, imp::PixelType::i32fC1>;
template class ImageGpu<imp::Pixel32fC2, imp::PixelType::i32fC2>;
template class ImageGpu<imp::Pixel32fC3, imp::PixelType::i32fC3>;
template class ImageGpu<imp::Pixel32fC4, imp::PixelType::i32fC4>;

} // namespace cu
              } // namespace imp

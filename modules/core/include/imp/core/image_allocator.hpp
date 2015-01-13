#ifndef IMP_IMAGE_ALLOCATOR_HPP
#define IMP_IMAGE_ALLOCATOR_HPP

//#include <cstring>
#include <math.h>

#include <imp/core/exception.hpp>

namespace iuprivate {

//--------------------------------------------------------------------------
template <typename PixelType>
class ImageAllocator
{
public:
  static PixelType* alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    //! @todo use sse malloc stuff so that pointers are aligned to 16/32-bytes! is there an optimal way to do that in windows and linux?

    if ((width == 0) || (height == 0)) throw IuException("width or height is 0", __FILE__,__FUNCTION__, __LINE__);

    // manually pitch the memory to 32-byte alignment (for better support of eg. IPP functions)
    *pitch = width * sizeof(PixelType);

    unsigned int elements_to_pitch = (32-(*pitch % 32))/sizeof(PixelType);

    // n*32 % 32 = 0 -> elements_to_pitch according to above formula would be (unnecessarily) 32 in that case
    // alternative formula: elements_to_pitch = ( 31 - ( ((*pitch) - 1) % 32) ) / sizeof(PixelType);
    if(*pitch % 32 == 0)
      elements_to_pitch = 0;

    width += elements_to_pitch;
    PixelType *buffer = new PixelType[width * height];
    *pitch = width * sizeof(PixelType);
    return buffer;
  }

  static void free(PixelType *buffer)
  {
    delete[] buffer;
  }

  static void copy(const PixelType *src, size_t src_pitch,
                   PixelType *dst, size_t dst_pitch, IuSize size)
  {
    size_t src_stride = src_pitch/sizeof(PixelType);
    size_t dst_stride = src_pitch/sizeof(PixelType);

    for(unsigned int y=0; y< size.height; ++y)
    {
      for(unsigned int x=0; x<size.width; ++x)
      {
        dst[y*dst_stride+x] = src[y*src_stride+x];
      }
    }
  }
};

//--------------------------------------------------------------------------
template <typename _PixelStorageType, int memaddr_align=32, bool align_rows=true>
class ImageMemoryStorage
{
public:
  ImageMemoryStorage() = default;
  virtual ~ImageMemoryStorage() = default;

  alloc(size_t num_elements)
  {
    if (num_elements == 0)
    {
      throw imp::Exception()
    }
  }

private:

};


} // namespace iuprivate

#endif // IMAGE_ALLOCATOR_CPU_H

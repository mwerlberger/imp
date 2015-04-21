#ifndef IMP_IMAGE_ALLOCATOR_HPP
#define IMP_IMAGE_ALLOCATOR_HPP

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <functional>
#include <algorithm>

#include <imp/core/exception.hpp>
#include <imp/core/size.hpp>

namespace imp {

//--------------------------------------------------------------------------
template <typename Pixel, int memaddr_align=32, bool align_rows=true>
struct MemoryStorage
{
public:
  typedef Pixel pixel_t;
  typedef pixel_t* pixel_container_t;
  typedef std::uint32_t size_type;

  //----------------------------------------------------------------------------
  MemoryStorage() = delete;
  virtual ~MemoryStorage() = delete;

  //----------------------------------------------------------------------------
  /**
   * @brief alignedAlloc allocates an aligned block of memory
   * @param num_elements Number of (minimum) allocated elements
   * @param init_with_zeros Flag if the memory elements should be zeroed out (default=false).
   *
   * @note Internally we use the C11 function aligned_alloc although there
   *       are also alignment functions in C++11 but aligned_alloc is the only
   *       one where we don't have to mess around with allocated bigger chungs of
   *       memory and shifting the start address accordingly. If you know a
   *       better approach using e.g. std::align(), let me know.
   */
  static pixel_container_t alignedAlloc(const size_t num_elements,
                                        bool init_with_zeros=false)
  {
    if (num_elements == 0)
    {
      throw imp::Exception("Failed to allocate memory: num_elements=0");
    }

    // restrict the memory address alignment to be in the interval ]0,128] and
    // of power-of-two using the 'complement and compare' method
    assert((memaddr_align != 0) && memaddr_align <= 128 &&
           ((memaddr_align & (~memaddr_align + 1)) == memaddr_align));

    const size_type memory_size = sizeof(pixel_t) * num_elements;
    //std::cout << "memory_size=" << memory_size << "; sizeof(pixel_t)=" << sizeof(pixel_t) << std::endl;

    pixel_container_t p_data_aligned =
        (pixel_container_t)aligned_alloc(memaddr_align, memory_size);

    if (p_data_aligned == nullptr)
    {
      throw std::bad_alloc();
    }

    if (init_with_zeros)
    {
      std::fill(p_data_aligned, p_data_aligned+num_elements, pixel_t(0));
    }

    return (pixel_container_t)p_data_aligned;
  }

  //----------------------------------------------------------------------------
  /**
   * @brief alignedAlloc allocates an aligned block that guarantees to host the image of size \a width \a x \a height
   * @param width Image width
   * @param height Image height
   * @param init_with_zeros Flag if the memory elements should be zeroed out (default=false).
   *
   * @note The allocator ensures that the starting adress of every row is aligned
   *       accordingly.
   *
   */
  static pixel_container_t alignedAlloc(const std::uint32_t width, const std::uint32_t height,
                                        size_type* pitch, bool init_with_zeros=false)
  {
    if (width == 0 || height == 0)
    {
      throw imp::Exception("Failed to allocate memory: width or height is zero");
    }

    // restrict the memory address alignment to be in the interval ]0,128] and
    // of power-of-two using the 'complement and compare' method
    assert((memaddr_align != 0) && memaddr_align <= 128 &&
           ((memaddr_align & (~memaddr_align + 1)) == memaddr_align));

    // check if the width allows a correct alignment of every row, otherwise add padding
    const size_type width_bytes = width * sizeof(pixel_t);
    // bytes % memaddr_align = 0 for bytes=n*memaddr_align is the reason for
    // the decrement in the following compution:
    const size_type bytes_to_add = (memaddr_align-1) - ((width_bytes-1) % memaddr_align);
    const std::uint32_t pitched_width = width + bytes_to_add/sizeof(pixel_t);
    *pitch = width_bytes + bytes_to_add;
    return alignedAlloc(pitched_width*height, init_with_zeros);
  }

  //----------------------------------------------------------------------------
  /**
   * @brief alignedAlloc allocates an aligned block of memory that guarantees to host the image of size \a size
   * @param size Image size
   * @param pitch Row alignment [bytes] if padding is needed.
   * @param init_with_zeros Flag if the memory elements should be zeroed out (default=false).
   * @return
   */
  static pixel_container_t alignedAlloc(imp::Size2u size, size_type* pitch,
                                       bool init_with_zeros=false)
  {
    return alignedAlloc(size[0], size[1], pitch, init_with_zeros);
  }

  //----------------------------------------------------------------------------
  static void free(pixel_container_t buffer)
  {
    free(buffer);
  }

};

//----------------------------------------------------------------------------
/**
 * @brief The Deallocator struct offers the ability to have custom deallocation methods.
 *
 * The Deallocator struct can be used as e.g. having custom deallocations with
 * shared pointers. Furthermore it enables the usage of external memory buffers
 * using shared pointers but not taking ownership of the memory. Be careful when
 * doing so as an application would behave badly if the memory got deleted although
 * we are still using it.
 *
 */
template<typename PixelStorageType>
struct MemoryDeallocator
{
  typedef PixelStorageType pixel_storage_t;
  typedef pixel_storage_t* pixel_container_t;

  // Default custom deleter assuming we use arrays (new PixelType[length])
  MemoryDeallocator()
    : f([](pixel_container_t p) { free(p); })
  { }

  // allow us to define a custom deallocator
  explicit MemoryDeallocator(std::function<void(pixel_container_t)> const &_f)
    : f(_f)
  { }

  void operator()(pixel_container_t p) const
  {
    f(p);
  }

private:
  std::function< void(pixel_container_t )> f;
};

} // imp

#endif // IMAGE_ALLOCATOR_HPP

#ifndef IMP_CU_IMAGE_ALLOCATOR_CUH
#define IMP_CU_IMAGE_ALLOCATOR_CUH

#include <iostream>

#include <cuda_runtime_api.h>

#include <imp/core/pixel_enums.hpp>
#include <imp/core/size.hpp>
#include <imp/cucore/cu_exception.hpp>

namespace imp { namespace cu {

//--------------------------------------------------------------------------
template <typename Pixel, imp::PixelType pixel_type = imp::PixelType::undefined>
struct MemoryStorage
{
public:
  typedef Pixel pixel_t;
  typedef pixel_t* pixel_container_t;
  typedef std::uint32_t size_type;

  //----------------------------------------------------------------------------
  // we don't construct an instance but only use it for the static functions
  // (at least for the moment)
  MemoryStorage() = delete;
  virtual ~MemoryStorage() = delete;

  //----------------------------------------------------------------------------
  //! @todo (MWE) do we wanna have a init flag for device memory?
  static pixel_container_t alloc(const size_t num_elements)
  {
    if (num_elements == 0)
    {
      throw imp::Exception("Failed to allocate memory: num_elements=0");
    }

    const size_type memory_size = sizeof(pixel_t) * num_elements;
    std::cout << "cu::MemoryStorage::alloc: memory_size=" << memory_size << "; sizeof(pixel_t)=" << sizeof(pixel_t) << std::endl;

    pixel_container_t p_data = nullptr;
    cudaError_t cu_err = cudaMalloc((void**)&p_data, memory_size);

    if (cu_err == cudaErrorMemoryAllocation)
    {
      throw std::bad_alloc();
    }
    else if (cu_err != cudaSuccess)
    {
      throw imp::cu::Exception("CUDA memory allocation failed", cu_err, __FILE__, __FUNCTION__, __LINE__);
    }

    return p_data;
  }

  //----------------------------------------------------------------------------
  /**
   * @brief alignedAlloc allocates an aligned 2D memory block (\a width x \a height) on the GPU (CUDA)
   * @param width Image width
   * @param height Image height
   * @param pitch Row alignment [bytes]
   *
   */
  static pixel_container_t alignedAlloc(const std::uint32_t width, const std::uint32_t height,
                                        size_type* pitch)
  {
    if (width == 0 || height == 0)
    {
      throw imp::cu::Exception("Failed to allocate memory: width or height is zero");
    }




    size_t width_bytes = width * sizeof(pixel_t);
    const int align_bytes = 4;
    if (pixel_type == imp::PixelType::i8uC3 && width_bytes % align_bytes)
    {
      width_bytes += (align_bytes-(width_bytes%align_bytes));
    }

    size_t intern_pitch;
    pixel_container_t p_data = nullptr;
    cudaError_t cu_err = cudaMallocPitch((void **)&p_data, &intern_pitch,
                                         width_bytes, (size_t)height);

    *pitch = intern_pitch;

    printf("pitch: %lu, i_pitch: %lu, width_bytes: %lu\n", *pitch, intern_pitch, width_bytes);

    if (cu_err == cudaErrorMemoryAllocation)
    {
      throw std::bad_alloc();
    }
    else if (cu_err != cudaSuccess)
    {
      throw imp::cu::Exception("CUDA memory allocation failed", cu_err, __FILE__, __FUNCTION__, __LINE__);
    }

    return p_data;
  }

  //----------------------------------------------------------------------------
  /**
   * @brief alignedAlloc allocates an aligned 2D memory block of given \a size on the GPU (CUDA)
   * @param size Image size
   * @param pitch Row alignment [bytes]
   * @return
   */
  static pixel_container_t alignedAlloc(imp::Size2u size, size_type* pitch)
  {
    return alignedAlloc(size[0], size[1], pitch);
  }

  //----------------------------------------------------------------------------
  static void free(pixel_container_t buffer)
  {
    cudaFree(buffer);
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
    : f([](pixel_container_t p) { printf("freeing cuda memory\n"); cudaFree(p); })
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

} // namespace cu
} // namespace imp

#endif // IMAGE_ALLOCATOR_HPP

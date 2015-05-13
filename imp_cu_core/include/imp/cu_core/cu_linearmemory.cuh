#ifndef IMP_CU_LINEARHOSTMEMORY_CUH
#define IMP_CU_LINEARHOSTMEMORY_CUH

//#include <stdio.h>
//#include <assert.h>
//#include <cstdlib>
#include <memory>

#include <imp/core/linearmemory_base.hpp>
#include <imp/core/linearmemory.hpp>

#include <imp/core/pixel.hpp>
#include <imp/cu_core/cu_memory_storage.cuh>

namespace imp {
namespace cu {

//struct DeviceData
//{

//};


template<typename Pixel>
class LinearMemory : public LinearMemoryBase
{
public:
  using Memory = imp::cu::MemoryStorage<Pixel>;
  using Deallocator = imp::cu::MemoryDeallocator<Pixel>;

public:
  __host__ LinearMemory();
  virtual ~LinearMemory() = default;

  __host__ LinearMemory(const size_t& length);
  __host__ LinearMemory(const imp::cu::LinearMemory<Pixel>& from);
  __host__ LinearMemory(const imp::LinearMemory<Pixel>& from);
  __host__ LinearMemory(Pixel* host_data, const size_t& length,
                        bool use_ext_data_pointer = false);

  /**
   * @brief Returns a pointer to the device buffer.
   * @param[in] offset Offset of the pointer array.
   * @return Pointer to the device buffer.
   *
   * @note The pointer can be offset to position \a offset.
   *
   */
  Pixel* data();

  /** Returns a const pointer to the device buffer.
   * @param[in] offset Desired offset within the array.
   * @return Const pointer to the device buffer.
   */
  const Pixel* data() const;

//  /** Sets a certain value to all pixels in the data vector.
//   */
//  void setValue(const Pixel& value);

  /** Copy data to another device class instance.
   */
  void copyTo(imp::cu::LinearMemory<Pixel>& dst);

  /** Copy data to a host class instance.
   */
  void copyTo(imp::LinearMemory<Pixel>& dst);

  /** Copy data from a host class instance.
   */
  void copyFrom(imp::LinearMemory<Pixel>& dst);

//  //! @todo (MWE) operator= for copyTo/copyFrom?
//  LinearMem& operator=(Pixel rhs);

  /** Returns the total amount of bytes saved in the data buffer. */
  virtual size_t bytes() const override { return this->length()*sizeof(Pixel); }

  /** Returns the bit depth of the data pointer. */
  virtual std::uint8_t bitDepth() const override { return 8*sizeof(Pixel); }

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool isGpuMemory() const  override { return true; }

private:
  std::unique_ptr<Pixel, Deallocator> data_;
};

// convenience typedefs
// (sync with explicit template class instantiations at the end of the cpp file)
typedef LinearMemory<imp::Pixel8uC1> LinearMemory8uC1;
typedef LinearMemory<imp::Pixel8uC2> LinearMemory8uC2;
typedef LinearMemory<imp::Pixel8uC2> LinearMemory8uC3;
typedef LinearMemory<imp::Pixel8uC2> LinearMemory8uC4;

typedef LinearMemory<imp::Pixel16uC1> LinearMemory16uC1;
typedef LinearMemory<imp::Pixel16uC2> LinearMemory16uC2;
typedef LinearMemory<imp::Pixel16uC2> LinearMemory16uC3;
typedef LinearMemory<imp::Pixel16uC2> LinearMemory16uC4;

typedef LinearMemory<imp::Pixel32sC1> LinearMemory32sC1;
typedef LinearMemory<imp::Pixel32sC2> LinearMemory32sC2;
typedef LinearMemory<imp::Pixel32sC2> LinearMemory32sC3;
typedef LinearMemory<imp::Pixel32sC2> LinearMemory32sC4;

typedef LinearMemory<imp::Pixel32fC1> LinearMemory32fC1;
typedef LinearMemory<imp::Pixel32fC2> LinearMemory32fC2;
typedef LinearMemory<imp::Pixel32fC2> LinearMemory32fC3;
typedef LinearMemory<imp::Pixel32fC2> LinearMemory32fC4;

} // namespace cu
} // namespace imp

#endif // IMP_LINEARHOSTMEMORY_H

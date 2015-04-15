#ifndef IMP_LINEARHOSTMEMORY_H
#define IMP_LINEARHOSTMEMORY_H

#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <memory>

#include <imp/core/linearmemory_base.hpp>
#include <imp/core/memory_storage.hpp>
#include <imp/core/pixel.hpp>

namespace imp {

template<typename Pixel>
class LinearMemory : public LinearMemoryBase
{
public:
  typedef LinearMemory<Pixel> LinearMem;
  typedef imp::MemoryStorage<Pixel> Memory;
  typedef imp::MemoryDeallocator<Pixel> Deallocator;

  typedef Pixel pixel_t;
  typedef pixel_t* pixel_container_t;

  LinearMemory();
  virtual ~LinearMemory() = default;

  LinearMemory(const size_t& length);
  LinearMemory(const LinearMemory<Pixel>& from);
  LinearMemory(pixel_container_t host_data, const size_t& length,
               bool use_ext_data_pointer = false);

  /**
   * @brief Returns a pointer to the device buffer.
   * @param[in] offset Offset of the pointer array.
   * @return Pointer to the device buffer.
   *
   * @note The pointer can be offset to position \a offset.
   *
   */
  Pixel* data(int offset = 0);

  /** Returns a const pointer to the device buffer.
   * @param[in] offset Desired offset within the array.
   * @return Const pointer to the device buffer.
   */
  const Pixel* data(int offset = 0) const;

  /** Sets a certain value to all pixels in the data vector.
   */
  void setValue(const Pixel& value);

  /** Copy data to another class instance.
   */
  void copyTo(LinearMemory<Pixel>& dst);

  //! @todo (MWE) operator= for copyTo/copyFrom?
  LinearMem& operator=(pixel_t rhs);

  /** Returns the total amount of bytes saved in the data buffer. */
  virtual size_t bytes() const override { return this->length()*sizeof(pixel_t); }

  /** Returns the bit depth of the data pointer. */
  virtual std::uint8_t bitDepth() const override { return 8*sizeof(pixel_t); }

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool isGpuMemory() const  override { return false; }

  /** Pixel access with (idx). */
   pixel_t& operator()(size_t idx) {return *this->data(idx);}
   /** Pixel access with [idx]. */
   pixel_t& operator[](size_t idx) {return *this->data(idx);}

private:
  std::unique_ptr<pixel_t, Deallocator> data_;

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



} // namespace imp

#endif // IMP_LINEARHOSTMEMORY_H

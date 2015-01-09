#ifndef IMP_LINEARHOSTMEMORY_H
#define IMP_LINEARHOSTMEMORY_H

#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <memory>

#include "linearmemory_base.hpp"


namespace imp {

template<typename PixelType>
class LinearMemory : public LinearMemoryBase
{
public:
  LinearMemory();
  virtual ~LinearMemory()
  {
    // let's praise managed pointers (and their custom deleter functions ;) ).
  }

  LinearMemory(const size_t& length);
  LinearMemory(const LinearMemory<PixelType>& from);
  LinearMemory(PixelType* host_data, const size_t& length,
                   bool use_ext_data_pointer = false);

  /**
   * @brief Returns a pointer to the device buffer.
   * @param[in] offset Offset of the pointer array.
   * @return Pointer to the device buffer.
   *
   * @note The pointer can be offset to position \a offset.
   *
   */
  PixelType* data(int offset = 0);

  /** Returns a const pointer to the device buffer.
   * @param[in] offset Desired offset within the array.
   * @return Const pointer to the device buffer.
   */
  const PixelType* data(int offset = 0) const;

  /** Sets a certain value to all pixels in the data vector.
   */
  void setValue(const PixelType& value);

  /** Copy data to another class instance.
   */
  void copyTo(LinearMemory<PixelType>& dst);

  //! @todo (MWE) operator= for copyTo/copyFrom?
  LinearMemory<PixelType>& operator=(PixelType rhs);

  /** Returns the total amount of bytes saved in the data buffer. */
  virtual size_t bytes() const override { return this->length()*sizeof(PixelType); }

  /** Returns the bit depth of the data pointer. */
  virtual std::uint8_t bitDepth() const override { return 8*sizeof(PixelType); }

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool isGpuMemory() const  override { return false; }

protected:

private:

  /** Custom deleter for the unique_ptr housing the c-style data pointer.
   * We do that in case we receive an external data pointer array but are not
   * allowed to manage the memory. In this case it is possible to avoid the
   * deletion of the array if the unique_ptr goes out of scope. Note that this is
   * a bit hacky but the most comfy solution for us internally. If you have a better
   * idea you can send me feedback on github (https://github.com/mwerlberger/imp).
   */
  struct CustomDataDeleter
  {
    // Default custom deleter assuming we use arrays (new PixelType[length])
    CustomDataDeleter()
      : f( [](PixelType* p) { delete[] p;} )
    {}

    // allow us to define a custom deleter
    explicit CustomDataDeleter(std::function< void(PixelType*)> const &f_ )
      : f(f_)
    {}

    void operator()(PixelType* p) const
    {
      f(p);
    }

  private:
    std::function< void(PixelType* )> f;
  };

  //PixelType* data_ = nullptr; /**< Pointer to device buffer. */
  std::unique_ptr<PixelType, CustomDataDeleter> data_;
  bool ext_data_pointer_ = false; /**< Flag for the ownership of the data pointer. */

};

// for explicit instantiation of the template class
typedef LinearMemory<std::uint8_t> LinearMemory_8u_C1;
typedef LinearMemory<std::uint16_t> LinearMemory_16u_C1;
typedef LinearMemory<std::int32_t> LinearMemory_32s_C1;
typedef LinearMemory<float> LinearMemory_32f_C1;



} // namespace imp

#endif // IMP_LINEARHOSTMEMORY_H

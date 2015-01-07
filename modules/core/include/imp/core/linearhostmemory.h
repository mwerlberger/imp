#ifndef IMP_LINEARHOSTMEMORY_H
#define IMP_LINEARHOSTMEMORY_H

#include <stdio.h>
#include <assert.h>
#include <cstdlib>

#include "linearmemory.h"


namespace imp {

template<typename PixelType>
class LinearHostMemory : public LinearMemory
{
public:
  LinearHostMemory();
  virtual ~LinearHostMemory()
  {
    if((!ext_data_pointer_) && (data_!=NULL))
    {
      delete[](data_);
      data_ = 0;
    }
  }

  LinearHostMemory(const size_t& length);
  LinearHostMemory(const LinearHostMemory<PixelType>& from);
  LinearHostMemory(PixelType* host_data, const size_t& length,
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
  void copyTo(LinearHostMemory<PixelType>& dst);

  //! @todo (MWE) operator= for copyTo/copyFrom?
  LinearHostMemory<PixelType>& operator=(PixelType rhs);

  /** Returns the total amount of bytes saved in the data buffer. */
  virtual size_t bytes() const { return this->length()*sizeof(PixelType); }

  /** Returns the bit depth of the data pointer. */
  virtual std::uint8_t bitDepth() const { return 8*sizeof(PixelType); }

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool onDevice() const { return false; }

protected:

private:
  PixelType* data_; /**< Pointer to device buffer. */
  bool ext_data_pointer_; /**< Flag for the ownership of the data pointer. */

};

// for explicit instantiation of the template class
typedef LinearHostMemory<std::uint8_t> LinearHostMemory_8u_C1;
typedef LinearHostMemory<std::uint16_t> LinearHostMemory_16u_C1;
typedef LinearHostMemory<std::int32_t> LinearHostMemory_32s_C1;
typedef LinearHostMemory<float> LinearHostMemory_32f_C1;



} // namespace imp

#endif // IMP_LINEARHOSTMEMORY_H

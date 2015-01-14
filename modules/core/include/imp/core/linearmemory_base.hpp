#ifndef IMP_LINEARMEMORY_HPP
#define IMP_LINEARMEMORY_HPP

#include "globaldefs.h"
//#include "coredefs.h"

namespace imp {

/** \brief LinearMemory Base class for linear memory classes.
  */
class LinearMemoryBase
{
protected:
  LinearMemoryBase() :
    length_(0)
  { }

  LinearMemoryBase(const LinearMemoryBase& from) :
    length_(from.length_)
  { }

  LinearMemoryBase(const size_t& length) :
    length_(length)
  { }

public:
  virtual ~LinearMemoryBase()
  { }

  /** Returns the number of elements saved in the device buffer. (length of device buffer) */
  size_t length() const
  {
    return length_;
  }

  /** Returns the total amount of bytes saved in the data buffer. */
  virtual size_t bytes() const = 0;

  /** Returns the bit depth of the data pointer. */
  virtual std::uint8_t bitDepth() const = 0;

  /** Returns flag if the image data resides on the GPU (TRUE) or CPU (FALSE) */
  virtual bool isGpuMemory() const = 0;

private:
  size_t length_;

};

} // namespace imp

#endif // IMP_LINEARMEMORY_HPP

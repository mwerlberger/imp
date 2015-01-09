#ifndef IMP_LINEARMEMORY_H
#define IMP_LINEARMEMORY_H


#include "globaldefs.h"
#include "coredefs.h"

namespace imp {

/** \brief LinearMemory Base class for linear memory classes.
  */
class LinearMemory
{
public:
  LinearMemory() :
    length_(0)
  { }

  LinearMemory(const LinearMemory& from) :
    length_(from.length_)
  { }

  LinearMemory(const size_t& length) :
    length_(length)
  { }

  virtual ~LinearMemory()
  { }

  /** Returns the number of elements saved in the device buffer. (length of device buffer) */
  size_t length() const
  {
    return length_;
  }

  //! @todo (MWE) don't we want to have these pure virtual?
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

#endif // IMP_LINEARMEMORY_H

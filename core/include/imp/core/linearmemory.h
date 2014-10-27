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

  LinearMemory(const unsigned int& length) :
    length_(length)
  { }

  virtual ~LinearMemory()
  { }

  /** Returns the number of elements saved in the device buffer. (length of device buffer) */
  unsigned int length() const
  {
    return length_;
  }

  /** Returns the total amount of bytes saved in the data buffer. */
  virtual size_t bytes() const {return 0;}

  /** Returns the bit depth of the data pointer. */
  virtual unsigned int bitDepth() const {return 0;}

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool onDevice() const {return false;}

private:
  unsigned int length_;


};

} // namespace imp

#endif // IMP_LINEARMEMORY_H

#include <imp/core/linearhostmemory.h>

#include <cstring>
#include <algorithm>


namespace imp {

//-----------------------------------------------------------------------------
template<typename PixelType>
LinearHostMemory<PixelType>::LinearHostMemory()
  : LinearMemory()
  , data_(0)
  , ext_data_pointer_(false)
{
}

//-----------------------------------------------------------------------------
template<typename PixelType>
LinearHostMemory<PixelType>::LinearHostMemory(const size_t& length)
  : LinearMemory(length)
  , data_(0)
  , ext_data_pointer_(false)
{
  data_ = new PixelType[this->length()];
}

//-----------------------------------------------------------------------------
template<typename PixelType>
LinearHostMemory<PixelType>::LinearHostMemory(const LinearHostMemory<PixelType>& from)
  : LinearMemory(from)
  , data_(0)
  , ext_data_pointer_(false)
{
  if (from.data_ == 0)
  {
    throw IuException("input data not valid", __FILE__, __FUNCTION__, __LINE__);
  }
  data_ = new PixelType[this->length()];
  std::copy(from.data_, from.data_+from.length(), data_);
}

//-----------------------------------------------------------------------------
template<typename PixelType>
LinearHostMemory<PixelType>::LinearHostMemory(PixelType* host_data,
                                              const size_t& length,
                                              bool use_ext_data_pointer)
  : LinearMemory(length)
  , data_(0)
  , ext_data_pointer_(use_ext_data_pointer)
{
  if (host_data == 0)
  {
    throw IuException("input data not valid", __FILE__, __FUNCTION__, __LINE__);
  }

  if(ext_data_pointer_)
  {
    // This uses the external data pointer and stores it as a 'reference' -> memory won't be managed by us!
    data_ = host_data;
  }
  else
  {
    // allocates an internal data pointer and copies the external data it.
    data_ = (PixelType*)malloc(this->length()*sizeof(PixelType));
    if (data_ == 0) throw std::bad_alloc();

    std::copy(host_data, host_data+length, data_);
  }
}


//-----------------------------------------------------------------------------
template<typename PixelType>
PixelType* LinearHostMemory<PixelType>::data(int offset)
{
  if ((size_t)offset > this->length())
  {
    throw IuException("offset not in range", __FILE__, __FUNCTION__, __LINE__);
  }

  return &(data_[offset]);
}

//-----------------------------------------------------------------------------
template<typename PixelType>
const PixelType* LinearHostMemory<PixelType>::data(int offset) const
{
  if ((size_t)offset > this->length())
  {
    throw IuException("offset not in range", __FILE__, __FUNCTION__, __LINE__);
  }
  return reinterpret_cast<const PixelType*>(&(data_[offset]));
}

//-----------------------------------------------------------------------------
template<typename PixelType>
void LinearHostMemory<PixelType>::setValue(const PixelType& value)
{
  std::fill(data_, data_+length(), value);
}

//=============================================================================
// Explicitely instantiate the desired classes
template class LinearHostMemory<unsigned char>;
template class LinearHostMemory<unsigned short>;
template class LinearHostMemory<float>;
template class LinearHostMemory<int>;


} // namespace imp

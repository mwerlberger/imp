#include <imp/core/linearhostmemory.h>

#include <cstring>
#include <algorithm>


namespace imp {

//-----------------------------------------------------------------------------
template<typename PixelType>
LinearHostMemory<PixelType>::LinearHostMemory()
  : LinearMemory()
{
}

//-----------------------------------------------------------------------------
template<typename PixelType>
LinearHostMemory<PixelType>::LinearHostMemory(const size_t& length)
  : LinearMemory(length)
  , data_(new PixelType[this->length()])
{
}

//-----------------------------------------------------------------------------
template<typename PixelType>
LinearHostMemory<PixelType>::LinearHostMemory(const LinearHostMemory<PixelType>& from)
  : LinearMemory(from)
{
  if (from.data_ == 0)
  {
    throw IuException("input data not valid", __FILE__, __FUNCTION__, __LINE__);
  }
  data_.reset(new PixelType[this->length()]);
  std::copy(from.data_.get(), from.data_.get()+from.length(), data_.get());
}

//-----------------------------------------------------------------------------
template<typename PixelType>
LinearHostMemory<PixelType>::LinearHostMemory(PixelType* host_data,
                                              const size_t& length,
                                              bool use_ext_data_pointer)
  : LinearMemory(length)
{
  if (host_data == 0)
  {
    throw IuException("input data not valid", __FILE__, __FUNCTION__, __LINE__);
  }

  if(ext_data_pointer_)
  {
    // This uses the external data pointer and stores it as a 'reference' -> memory won't be managed by us!
    auto no_delete_fcn = [](PixelType*) {};
    data_ = std::unique_ptr<PixelType, CustomDataDeleter>(
          host_data, CustomDataDeleter(no_delete_fcn));
  }
  else
  {
    // allocates an internal data pointer and copies the external data it.
    data_ = std::unique_ptr<PixelType, CustomDataDeleter>(new PixelType[this->length()]);
    std::copy(host_data, host_data+length, data_.get());
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

  return &(data_.get()[offset]);
}

//-----------------------------------------------------------------------------
template<typename PixelType>
const PixelType* LinearHostMemory<PixelType>::data(int offset) const
{
  if ((size_t)offset > this->length())
  {
    throw IuException("offset not in range", __FILE__, __FUNCTION__, __LINE__);
  }
  return reinterpret_cast<const PixelType*>(&(data_.get()[offset]));
}

//-----------------------------------------------------------------------------
template<typename PixelType>
void LinearHostMemory<PixelType>::setValue(const PixelType& value)
{
  std::fill(data_.get(), data_.get()+this->length(), value);
}

//-----------------------------------------------------------------------------
template<typename PixelType>
void LinearHostMemory<PixelType>::copyTo(LinearHostMemory<PixelType>& dst)
{
  if (this->length() != dst.length())
  {
    throw IuException("source and destination array are of different length", __FILE__, __FUNCTION__, __LINE__);
  }
  std::copy(data_.get(), data_.get()+this->length(), dst.data_.get());
}


//-----------------------------------------------------------------------------
template<typename PixelType>
LinearHostMemory<PixelType>& LinearHostMemory<PixelType>::operator=(PixelType rhs)
{
  this->setValue(rhs);
  return *this;
}


//=============================================================================
// Explicitely instantiate the desired classes
template class LinearHostMemory<std::uint8_t>;
template class LinearHostMemory<std::uint16_t>;
template class LinearHostMemory<std::int32_t>;
template class LinearHostMemory<float>;

} // namespace imp

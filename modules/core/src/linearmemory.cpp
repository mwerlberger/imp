#include <imp/core/linearmemory.hpp>

#include <cstring>
#include <algorithm>

#include <imp/core/exception.hpp>


namespace imp {

//-----------------------------------------------------------------------------
template<typename Pixel>
LinearMemory<Pixel>::LinearMemory()
  : LinearMemoryBase()
{
}

//-----------------------------------------------------------------------------
template<typename Pixel>
LinearMemory<Pixel>::LinearMemory(const size_t& length)
  : LinearMemoryBase(length)
  , data_(Memory::alignedAlloc(this->length()))
{
}

//-----------------------------------------------------------------------------
template<typename Pixel>
LinearMemory<Pixel>::LinearMemory(const LinearMemory<Pixel>& from)
  : LinearMemoryBase(from)
{
  if (from.data_ == 0)
  {
    throw imp::Exception("input data not valid", __FILE__, __FUNCTION__, __LINE__);
  }
  data_.reset(Memory::alignedAlloc(this->length()));
  std::copy(from.data_.get(), from.data_.get()+from.length(), data_.get());
}

//-----------------------------------------------------------------------------
template<typename Pixel>
LinearMemory<Pixel>::LinearMemory(pixel_container_t host_data,
                                  const size_t& length,
                                  bool use_ext_data_pointer)
  : LinearMemoryBase(length)
{
  if (host_data == nullptr)
  {
    throw imp::Exception("input data not valid", __FILE__, __FUNCTION__, __LINE__);
  }

  if(use_ext_data_pointer)
  {
    // This uses the external data pointer and stores it as a 'reference':
    // memory won't be managed by us!
    auto dealloc_nop = [](pixel_container_t) { ; };
    data_ = std::unique_ptr<pixel_t, Deallocator>(
          host_data, Deallocator(dealloc_nop));
  }
  else
  {
    // allocates an internal data pointer and copies the external data it.
    data_.reset(Memory::alignedAlloc(this->length()));
    std::copy(host_data, host_data+length, data_.get());
  }
}

//-----------------------------------------------------------------------------
template<typename Pixel>
Pixel* LinearMemory<Pixel>::data(int offset)
{
  if ((size_t)offset > this->length())
  {
    throw imp::Exception("offset not in range", __FILE__, __FUNCTION__, __LINE__);
  }

  return &(data_.get()[offset]);
}

//-----------------------------------------------------------------------------
template<typename Pixel>
const Pixel* LinearMemory<Pixel>::data(int offset) const
{
  if ((size_t)offset > this->length())
  {
    throw imp::Exception("offset not in range", __FILE__, __FUNCTION__, __LINE__);
  }
  return reinterpret_cast<const Pixel*>(&(data_.get()[offset]));
}

//-----------------------------------------------------------------------------
template<typename Pixel>
void LinearMemory<Pixel>::setValue(const Pixel& value)
{
  std::fill(data_.get(), data_.get()+this->length(), value);
}

//-----------------------------------------------------------------------------
template<typename Pixel>
void LinearMemory<Pixel>::copyTo(LinearMemory<Pixel>& dst)
{
  if (this->length() != dst.length())
  {
    throw imp::Exception("source and destination array are of different length", __FILE__, __FUNCTION__, __LINE__);
  }
  std::copy(data_.get(), data_.get()+this->length(), dst.data_.get());
}


//-----------------------------------------------------------------------------
template<typename Pixel>
LinearMemory<Pixel>& LinearMemory<Pixel>::operator=(Pixel rhs)
{
  this->setValue(rhs);
  return *this;
}


//=============================================================================
// Explicitely instantiate the desired classes
template class LinearMemory<imp::Pixel8uC1>;
template class LinearMemory<imp::Pixel8uC2>;
template class LinearMemory<imp::Pixel16uC1>;
template class LinearMemory<imp::Pixel32sC1>;
template class LinearMemory<imp::Pixel32fC1>;

} // namespace imp

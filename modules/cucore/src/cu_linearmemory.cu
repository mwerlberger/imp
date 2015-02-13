#include <imp/cucore/cu_linearmemory.cuh>

//#include <cstring>
//#include <algorithm>

#include <imp/cucore/cu_exception.hpp>


namespace imp {
namespace cu {

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
  , data_(CuMemory::alloc(this->length()))
{
}

//-----------------------------------------------------------------------------
template<typename Pixel>
LinearMemory<Pixel>::LinearMemory(const imp::cu::LinearMemory<Pixel>& from)
  : imp::LinearMemoryBase(from)
{
  if (from.data_ == 0)
  {
    throw imp::cu::Exception("'from' data not valid", __FILE__, __FUNCTION__, __LINE__);
  }

  data_.reset(CuMemory::alloc(this->length()));
  const cudaError cu_err =
      cudaMemcpy(data_.get(), from.data(), this->bytes(), cudaMemcpyDeviceToDevice);

  if (cu_err != cudaSuccess)
  {
    throw imp::cu::Exception("cudaMemcpy returned error code", cu_err, __FILE__, __FUNCTION__, __LINE__);
  }
}

//-----------------------------------------------------------------------------
template<typename Pixel>
LinearMemory<Pixel>::LinearMemory(const imp::LinearMemory<Pixel>& from)
  : imp::LinearMemoryBase(from)
{
  if (from.data() == 0)
  {
    throw imp::cu::Exception("'from' data not valid", __FILE__, __FUNCTION__, __LINE__);
  }

  data_.reset(CuMemory::alloc(this->length()));
  const cudaError cu_err =
      cudaMemcpy(data_.get(), from.data(), this->bytes(), cudaMemcpyHostToDevice);

  if (cu_err != cudaSuccess)
  {
    throw imp::cu::Exception("cudaMemcpy returned error code", cu_err, __FILE__, __FUNCTION__, __LINE__);
  }
}

////-----------------------------------------------------------------------------
//template<typename Pixel>
//LinearMemory<Pixel>::LinearMemory(pixel_container_t host_data,
//                                  const size_t& length,
//                                  bool use_ext_data_pointer)
//  : LinearMemoryBase(length)
//{
//  if (host_data == nullptr)
//  {
//    throw imp::cu::Exception("input data not valid", __FILE__, __FUNCTION__, __LINE__);
//  }

//  if(use_ext_data_pointer)
//  {
//    // This uses the external data pointer and stores it as a 'reference':
//    // memory won't be managed by us!
//    auto dealloc_nop = [](pixel_container_t) { ; };
//    data_ = std::unique_ptr<pixel_t, Deallocator>(
//          host_data, Deallocator(dealloc_nop));
//  }
//  else
//  {
//    // allocates an internal data pointer and copies the external data it.
//    data_.reset(CuMemory::alignedAlloc(this->length()));
//    std::copy(host_data, host_data+length, data_.get());
//  }
//}

//-----------------------------------------------------------------------------
template<typename Pixel>
Pixel* LinearMemory<Pixel>::data()
{
  return data_.get();
}

//-----------------------------------------------------------------------------
template<typename Pixel>
const Pixel* LinearMemory<Pixel>::data() const
{
  return reinterpret_cast<const Pixel*>(data_.get());
}

////-----------------------------------------------------------------------------
//template<typename Pixel>
//void LinearMemory<Pixel>::setValue(const Pixel& value)
//{
//  std::fill(data_.get(), data_.get()+this->length(), value);
//}

//-----------------------------------------------------------------------------
template<typename Pixel>
void LinearMemory<Pixel>::copyTo(imp::cu::LinearMemory<Pixel>& dst)
{
  if (this->bytes() != dst.bytes())
  {
    throw imp::cu::Exception("source and destination array are of different length (byte length checked)", __FILE__, __FUNCTION__, __LINE__);
  }

  const cudaError cu_err =
      cudaMemcpy(dst.data(), this->data(), this->bytes(), cudaMemcpyDeviceToDevice);

  if (cu_err != cudaSuccess)
  {
    throw imp::cu::Exception("cudaMemcpy returned error code", cu_err, __FILE__, __FUNCTION__, __LINE__);
  }
}

//-----------------------------------------------------------------------------
template<typename Pixel>
void LinearMemory<Pixel>::copyTo(imp::LinearMemory<Pixel>& dst)
{
  if (this->bytes() != dst.bytes())
  {
    throw imp::cu::Exception("source and destination array are of different length (byte length checked)", __FILE__, __FUNCTION__, __LINE__);
  }

  const cudaError cu_err =
      cudaMemcpy(dst.data(), this->data(), this->bytes(), cudaMemcpyDeviceToHost);

  if (cu_err != cudaSuccess)
  {
    throw imp::cu::Exception("cudaMemcpy returned error code", cu_err, __FILE__, __FUNCTION__, __LINE__);
  }
}


//-----------------------------------------------------------------------------
template<typename Pixel>
void LinearMemory<Pixel>::copyFrom(imp::LinearMemory<Pixel>& from)
{
  if (this->bytes() != from.bytes())
  {
    throw imp::cu::Exception("source and destination array are of different length (byte length checked)", __FILE__, __FUNCTION__, __LINE__);
  }

  const cudaError cu_err =
      cudaMemcpy(this->data(), from.data(), this->bytes(), cudaMemcpyHostToDevice);

  if (cu_err != cudaSuccess)
  {
    throw imp::cu::Exception("cudaMemcpy returned error code", cu_err, __FILE__, __FUNCTION__, __LINE__);
  }
}


////-----------------------------------------------------------------------------
//template<typename Pixel>
//LinearMemory<Pixel>& LinearMemory<Pixel>::operator=(Pixel rhs)
//{
//  this->setValue(rhs);
//  return *this;
//}


//=============================================================================
// Explicitely instantiate the desired classes
template class LinearMemory<imp::Pixel8uC1>;
template class LinearMemory<imp::Pixel8uC2>;
template class LinearMemory<imp::Pixel8uC3>;
template class LinearMemory<imp::Pixel8uC4>;

template class LinearMemory<imp::Pixel16uC1>;
template class LinearMemory<imp::Pixel16uC2>;
template class LinearMemory<imp::Pixel16uC3>;
template class LinearMemory<imp::Pixel16uC4>;

template class LinearMemory<imp::Pixel32sC1>;
template class LinearMemory<imp::Pixel32sC2>;
template class LinearMemory<imp::Pixel32sC3>;
template class LinearMemory<imp::Pixel32sC4>;

template class LinearMemory<imp::Pixel32fC1>;
template class LinearMemory<imp::Pixel32fC2>;
template class LinearMemory<imp::Pixel32fC3>;
template class LinearMemory<imp::Pixel32fC4>;

} // namespace cu
} // namespace imp

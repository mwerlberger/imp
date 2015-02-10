#ifndef IMP_CU_PIXEL_CONVERSION_HPP
#define IMP_CU_PIXEL_CONVERSION_HPP

#include <cuda_runtime_api.h>
#include <imp/core/pixel.hpp>
#include <imp/core/pixel_enums.hpp>
#include <imp/cucore/cu_image_gpu.cuh>

namespace imp {
namespace cu {

//
// uchar
//
uchar1* __host__ __device__ toCudaVectorType(imp::Pixel8uC1* buffer)
{
  return reinterpret_cast<uchar1*>(buffer);
}
uchar2* __host__ __device__ toCudaVectorType(imp::Pixel8uC2* buffer)
{
  return reinterpret_cast<uchar2*>(buffer);
}
uchar3* __host__ __device__ toCudaVectorType(imp::Pixel8uC3* buffer)
{
  return reinterpret_cast<uchar3*>(buffer);
}
uchar4* __host__ __device__ toCudaVectorType(imp::Pixel8uC4* buffer)
{
  return reinterpret_cast<uchar4*>(buffer);
}

//
// ushort
//
ushort1* __host__ __device__ toCudaVectorType(imp::Pixel16uC1* buffer)
{
  return reinterpret_cast<ushort1*>(buffer);
}
ushort2* __host__ __device__ toCudaVectorType(imp::Pixel16uC2* buffer)
{
  return reinterpret_cast<ushort2*>(buffer);
}
ushort3* __host__ __device__ toCudaVectorType(imp::Pixel16uC3* buffer)
{
  return reinterpret_cast<ushort3*>(buffer);
}
ushort4* __host__ __device__ toCudaVectorType(imp::Pixel16uC4* buffer)
{
  return reinterpret_cast<ushort4*>(buffer);
}

//
// int
//
int1* __host__ __device__ toCudaVectorType(imp::Pixel32sC1* buffer)
{
  return reinterpret_cast<int1*>(buffer);
}
int2* __host__ __device__ toCudaVectorType(imp::Pixel32sC2* buffer)
{
  return reinterpret_cast<int2*>(buffer);
}
int3* __host__ __device__ toCudaVectorType(imp::Pixel32sC3* buffer)
{
  return reinterpret_cast<int3*>(buffer);
}
int4* __host__ __device__ toCudaVectorType(imp::Pixel32sC4* buffer)
{
  return reinterpret_cast<int4*>(buffer);
}

//
// float
//
float1* __host__ __device__ toCudaVectorType(imp::Pixel32fC1* buffer)
{
  return reinterpret_cast<float1*>(buffer);
}
float2* __host__ __device__ toCudaVectorType(imp::Pixel32fC2* buffer)
{
  return reinterpret_cast<float2*>(buffer);
}
float3* __host__ __device__ toCudaVectorType(imp::Pixel32fC3* buffer)
{
  return reinterpret_cast<float3*>(buffer);
}
float4* __host__ __device__ toCudaVectorType(imp::Pixel32fC4* buffer)
{
  return reinterpret_cast<float4*>(buffer);
}


cudaChannelFormatDesc toCudaChannelFormatDesc(imp::PixelType pixel_type)
{
  switch (pixel_type)
  {
  case imp::PixelType::i8uC1:
  return cudaCreateChannelDesc<uchar1>();
  case imp::PixelType::i8uC2:
  return cudaCreateChannelDesc<uchar2>();
  case imp::PixelType::i8uC3:
  return cudaCreateChannelDesc<uchar3>();
  case imp::PixelType::i8uC4:
  return cudaCreateChannelDesc<uchar4>();
  case imp::PixelType::i16uC1:
  return cudaCreateChannelDesc<short1>();
  case imp::PixelType::i16uC2:
  return cudaCreateChannelDesc<short2>();
  case imp::PixelType::i16uC3:
  return cudaCreateChannelDesc<short3>();
  case imp::PixelType::i16uC4:
  return cudaCreateChannelDesc<short4>();
  case imp::PixelType::i32sC1:
  return cudaCreateChannelDesc<int1>();
  case imp::PixelType::i32sC2:
  return cudaCreateChannelDesc<int2>();
  case imp::PixelType::i32sC3:
  return cudaCreateChannelDesc<int3>();
  case imp::PixelType::i32sC4:
  return cudaCreateChannelDesc<int4>();
  case imp::PixelType::i32fC1:
  return cudaCreateChannelDesc<float1>();
  case imp::PixelType::i32fC2:
  return cudaCreateChannelDesc<float2>();
  case imp::PixelType::i32fC3:
  return cudaCreateChannelDesc<float3>();
  case imp::PixelType::i32fC4:
  return cudaCreateChannelDesc<float4>();
  default:
    throw imp::cu::Exception("Pixel type not supported to generate a CUDA texture.",
                             __FILE__, __FUNCTION__, __LINE__);
  }
}



} // namespace cu
} // namespace imp


#endif // IMP_CU_PIXEL_CONVERSION_HPP


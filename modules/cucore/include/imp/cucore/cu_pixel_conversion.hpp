#ifndef IMP_CU_PIXEL_CONVERSION_HPP
#define IMP_CU_PIXEL_CONVERSION_HPP

#include <cuda_runtime_api.h>
#include <imp/core/pixel.hpp>
#include <imp/core/pixel_enums.hpp>

namespace imp {
namespace cu {

//
// uchar
//
uchar1* __host__ __device__ toCudaVectorType(imp::Pixel8uC1* buffer) __attribute__ ((unused));
uchar2* __host__ __device__ toCudaVectorType(imp::Pixel8uC2* buffer) __attribute__ ((unused));
uchar3* __host__ __device__ toCudaVectorType(imp::Pixel8uC3* buffer) __attribute__ ((unused));
uchar4* __host__ __device__ toCudaVectorType(imp::Pixel8uC4* buffer) __attribute__ ((unused));

//
// ushort
//
ushort1* __host__ __device__ toCudaVectorType(imp::Pixel16uC1* buffer) __attribute__ ((unused));
ushort2* __host__ __device__ toCudaVectorType(imp::Pixel16uC2* buffer) __attribute__ ((unused));
ushort3* __host__ __device__ toCudaVectorType(imp::Pixel16uC3* buffer) __attribute__ ((unused));
ushort4* __host__ __device__ toCudaVectorType(imp::Pixel16uC4* buffer) __attribute__ ((unused));

//
// int
//
int1* __host__ __device__ toCudaVectorType(imp::Pixel32sC1* buffer) __attribute__ ((unused));
int2* __host__ __device__ toCudaVectorType(imp::Pixel32sC2* buffer) __attribute__ ((unused));
int3* __host__ __device__ toCudaVectorType(imp::Pixel32sC3* buffer) __attribute__ ((unused));
int4* __host__ __device__ toCudaVectorType(imp::Pixel32sC4* buffer) __attribute__ ((unused));

//
// float
//
float1* __host__ __device__ toCudaVectorType(imp::Pixel32fC1* buffer) __attribute__ ((unused));
float2* __host__ __device__ toCudaVectorType(imp::Pixel32fC2* buffer) __attribute__ ((unused));
float3* __host__ __device__ toCudaVectorType(imp::Pixel32fC3* buffer) __attribute__ ((unused));
float4* __host__ __device__ toCudaVectorType(imp::Pixel32fC4* buffer) __attribute__ ((unused));


cudaChannelFormatDesc toCudaChannelFormatDesc(imp::PixelType pixel_type) __attribute__ ((unused));



} // namespace cu
} // namespace imp


#endif // IMP_CU_PIXEL_CONVERSION_HPP


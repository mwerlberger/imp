#ifndef IMP_CU_TEXTURE_CUH
#define IMP_CU_TEXTURE_CUH

#include <cstring>
#include <cuda_runtime_api.h>
#include <imp/core/types.hpp>
#include <imp/core/pixel.hpp>
//#include <imp/core/pixel_enums.hpp>
//#include <imp/cucore/cu_image_gpu.cuh>
//#include <imp/cucore/cu_pixel_conversion.hpp>

namespace imp { namespace cu {

/**
 * @brief The Texture struct
 */
struct Texture
{
  cudaTextureObject_t tex_object;

  __host__ Texture() = default;
  __host__ virtual ~Texture() = default;
  __device__ __forceinline__ operator cudaTextureObject_t() const {return tex_object;}

};

/**
 * @brief The Texture2D struct
 */
struct Texture2D : Texture
{
  using Texture::Texture;
  __host__ Texture2D(void* data, size_type pitch,
                     cudaChannelFormatDesc channel_desc,
                     imp::Size2u size,
                     bool _normalized_coords = false,
                     cudaTextureFilterMode filter_mode = cudaFilterModePoint,
                     cudaTextureAddressMode address_mode = cudaAddressModeClamp,
                     cudaTextureReadMode read_mode = cudaReadModeElementType)
    : Texture()
  {
    cudaResourceDesc tex_res;
    std::memset(&tex_res, 0, sizeof(tex_res));
    tex_res.resType = cudaResourceTypePitch2D;
    tex_res.res.pitch2D.width = size.width();
    tex_res.res.pitch2D.height = size.height();
    tex_res.res.pitch2D.pitchInBytes = pitch;
    tex_res.res.pitch2D.devPtr = data;
    tex_res.res.pitch2D.desc = channel_desc;

    cudaTextureDesc tex_desc;
    std::memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.normalizedCoords = _normalized_coords;
    tex_desc.filterMode = filter_mode;
    tex_desc.addressMode[0] = address_mode;
    tex_desc.addressMode[1] = address_mode;
    tex_desc.readMode = read_mode;

    cudaCreateTextureObject(&(this->tex_object), &tex_res, &tex_desc, 0);

  }

  __host__ virtual ~Texture2D()
  {
    cudaDestroyTextureObject(this->tex_object);
  }


  /**
   * Wrapper for accesing texels. The coordinate are not texture but pixel coords (no need to add 0.5f!)
   */
  template<typename T>
  __device__ __forceinline__ T fetch(float x, float y) const
  {
    return tex2D<T>(tex_object, x+.5f, y+.5f);
  }

  /*
   *     FETCH OVERLOADS AS TEX2D ONLY ALLOWS FOR CUDA VEC TYPES
   */

  __device__ __forceinline__ void fetch(imp::Pixel8uC1& texel, float x, float y)
  {
    texel = imp::Pixel8uC1(this->fetch<uchar1>(x,y).x);
  }
  __device__ __forceinline__ void fetch(imp::Pixel8uC2& texel, float x, float y)
  {
    uchar2 val = this->fetch<uchar2>(x,y);
    texel = imp::Pixel8uC2(val.x, val.y);
  }
//  __device__ __forceinline__ void fetch(imp::Pixel8uC3& texel, float x, float y)
//  {
//    uchar3 val = this->fetch<uchar3>(x,y);
//    texel = imp::Pixel8uC3(val.x, val.y, val.z);
//  }
  __device__ __forceinline__ void fetch(imp::Pixel8uC4& texel, float x, float y)
  {
    uchar4 val = this->fetch<uchar4>(x,y);
    texel = imp::Pixel8uC4(val.x, val.y, val.z, val.w);
  }

  __device__ __forceinline__ void fetch(imp::Pixel16uC1& texel, float x, float y)
  {
    texel = imp::Pixel16uC1(this->fetch<ushort1>(x,y).x);
  }
  __device__ __forceinline__ void fetch(imp::Pixel16uC2& texel, float x, float y)
  {
    ushort2 val = this->fetch<ushort2>(x,y);
    texel = imp::Pixel16uC2(val.x, val.y);
  }
//  __device__ __forceinline__ void fetch(imp::Pixel16uC3& texel, float x, float y)
//  {
//    ushort3 val = this->fetch<ushort3>(x,y);
//    texel = imp::Pixel16uC3(val.x, val.y, val.z);
//  }
  __device__ __forceinline__ void fetch(imp::Pixel16uC4& texel, float x, float y)
  {
    ushort4 val = this->fetch<ushort4>(x,y);
    texel = imp::Pixel16uC4(val.x, val.y, val.z, val.w);
  }

  __device__ __forceinline__ void fetch(imp::Pixel32sC1& texel, float x, float y)
  {
    texel = imp::Pixel32sC1(this->fetch<int1>(x,y).x);
  }
  __device__ __forceinline__ void fetch(imp::Pixel32sC2& texel, float x, float y)
  {
    int2 val = this->fetch<int2>(x,y);
    texel = imp::Pixel32sC2(val.x, val.y);
  }
//  __device__ __forceinline__ void fetch(imp::Pixel32sC3& texel, float x, float y)
//  {
//    int3 val = this->fetch<int3>(x,y);
//    texel = imp::Pixel32sC3(val.x, val.y, val.z);
//  }
  __device__ __forceinline__ void fetch(imp::Pixel32sC4& texel, float x, float y)
  {
    int4 val = this->fetch<int4>(x,y);
    texel = imp::Pixel32sC4(val.x, val.y, val.z, val.w);
  }

  __device__ __forceinline__ void fetch(imp::Pixel32fC1& texel, float x, float y)
  {
    texel = imp::Pixel32fC1(this->fetch<float1>(x,y).x);
  }
  __device__ __forceinline__ void fetch(imp::Pixel32fC2& texel, float x, float y)
  {
    float2 val = this->fetch<float2>(x,y);
    texel = imp::Pixel32fC2(val.x, val.y);
  }
//  __device__ __forceinline__ void fetch(imp::Pixel32fC3& texel, float x, float y)
//  {
//    float3 val = this->fetch<float3>(x,y);
//    texel = imp::Pixel32fC3(val.x, val.y, val.z);
//  }
  __device__ __forceinline__ void fetch(imp::Pixel32fC4& texel, float x, float y)
  {
    float4 val = this->fetch<float4>(x,y);
    texel = imp::Pixel32fC4(val.x, val.y, val.z, val.w);
  }

};



} // namespace cu
} // namespace imp

#endif // IMP_CU_TEXTURE_CUH


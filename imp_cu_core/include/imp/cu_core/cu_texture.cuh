#ifndef IMP_CU_TEXTURE_CUH
#define IMP_CU_TEXTURE_CUH

#include <memory>
#include <cstring>
#include <cuda_runtime.h>
#include <imp/core/types.hpp>
#include <imp/core/pixel.hpp>
#include <imp/core/pixel_enums.hpp>
#include <imp/cu_core/cu_exception.hpp>

namespace imp {
namespace cu {

/**
 * @brief The Texture2D struct wrappes the cuda texture object
 */
struct Texture2D
{
  cudaTextureObject_t tex_object;
  __device__ __forceinline__ operator cudaTextureObject_t() const {return tex_object;}

  using Ptr = std::shared_ptr<Texture2D>;
  using UPtr = std::unique_ptr<Texture2D>;

  __host__ Texture2D() = delete;

  __host__ Texture2D(cudaTextureObject_t _tex_object)
    : tex_object(_tex_object)
  {

  }

  __host__ Texture2D(const void* data, size_type pitch,
                     cudaChannelFormatDesc channel_desc,
                     imp::Size2u size,
                     bool _normalized_coords = false,
                     cudaTextureFilterMode filter_mode = cudaFilterModePoint,
                     cudaTextureAddressMode address_mode = cudaAddressModeClamp,
                     cudaTextureReadMode read_mode = cudaReadModeElementType)
  {
    cudaResourceDesc tex_res;
    std::memset(&tex_res, 0, sizeof(tex_res));
    tex_res.resType = cudaResourceTypePitch2D;
    tex_res.res.pitch2D.width = size.width();
    tex_res.res.pitch2D.height = size.height();
    tex_res.res.pitch2D.pitchInBytes = pitch;
    tex_res.res.pitch2D.devPtr = const_cast<void*>(data);
    tex_res.res.pitch2D.desc = channel_desc;

    cudaTextureDesc tex_desc;
    std::memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.normalizedCoords = (_normalized_coords==true) ? 1 : 0;
    tex_desc.filterMode = filter_mode;
    tex_desc.addressMode[0] = address_mode;
    tex_desc.addressMode[1] = address_mode;
    tex_desc.readMode = read_mode;

    cudaError_t err = cudaCreateTextureObject(&tex_object, &tex_res, &tex_desc, 0);
    if  (err != ::cudaSuccess)
    {
      throw imp::cu::Exception("Failed to create texture object", err,
                               __FILE__, __FUNCTION__, __LINE__);
    }
  }

  __host__ virtual ~Texture2D()
  {
    cudaError_t err = cudaDestroyTextureObject(tex_object);
    if  (err != ::cudaSuccess)
    {
      throw imp::cu::Exception("Failed to destroy texture object", err,
                               __FILE__, __FUNCTION__, __LINE__);
    }
  }

  // copy and asignment operator
  __host__ __device__
  Texture2D(const Texture2D& other)
    : tex_object(other.tex_object)
  {
  }
  __host__ __device__
  Texture2D& operator=(const Texture2D& other)
  {
    if  (this != &other)
    {
      tex_object = other.tex_object;
    }
    return *this;
  }

  /**
   * @brief Wrapper for accesing texels including coord manipulation {e.g. x = (x+.5f)*mul_x + add_x}
   * @param mul_x multiplicative factor of x-coordinate
   * @param mul_y multiplicative factor of y-coordinate
   * @param add_x additive factor of x-coordinate
   * @param add_y additive factor of y-coordinate
   * @note The coordinate are not texture but pixel coords (no need to add 0.5f!)
   */
  template<typename T>
  __device__ __forceinline__ T fetch(float x, float y,
                                     float mul_x=1.0f, float mul_y=1.0f,
                                     float add_x=0.0f, float add_y=0.0f) const
  {
    return tex2D<T>(tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
  }

  /*
   *     FETCH OVERLOADS AS TEX2D ONLY ALLOWS FOR CUDA VEC TYPES
   */

  __device__ __forceinline__ void fetch(imp::Pixel8uC1& texel, float x,
                                        float y, float mul_x=1.0f, float mul_y=1.0f,
                                        float add_x=0.0f, float add_y=0.0f)
  {
    texel = imp::Pixel8uC1(this->fetch<uchar1>(x,y,mul_x,mul_y,add_x,add_y).x);
  }
  __device__ __forceinline__ void fetch(imp::Pixel8uC2& texel, float x, float y,
                                        float mul_x=1.0f, float mul_y=1.0f,
                                        float add_x=0.0f, float add_y=0.0f)
  {
    uchar2 val = this->fetch<uchar2>(x,y,mul_x,mul_y,add_x,add_y);
    texel = imp::Pixel8uC2(val.x, val.y);
  }
  //  __device__ __forceinline__ void fetch(imp::Pixel8uC3& texel, float x, float y,
  //                                        float mul_x=1.0f, float mul_y=1.0f,
  //                                        float add_x=0.0f, float add_y=0.0f)
  //  {
  //    uchar3 val = this->fetch<uchar3>(x,y,mul_x,mul_y,add_x,add_y);
  //    texel = imp::Pixel8uC3(val.x, val.y, val.z);
  //  }
  __device__ __forceinline__ void fetch(imp::Pixel8uC4& texel, float x, float y,
                                        float mul_x=1.0f, float mul_y=1.0f,
                                        float add_x=0.0f, float add_y=0.0f)
  {
    uchar4 val = this->fetch<uchar4>(x,y,mul_x,mul_y,add_x,add_y);
    texel = imp::Pixel8uC4(val.x, val.y, val.z, val.w);
  }

  __device__ __forceinline__ void fetch(imp::Pixel16uC1& texel, float x, float y,
                                        float mul_x=1.0f, float mul_y=1.0f,
                                        float add_x=0.0f, float add_y=0.0f)
  {
    texel = imp::Pixel16uC1(this->fetch<ushort1>(x,y,mul_x,mul_y,add_x,add_y).x);
  }
  __device__ __forceinline__ void fetch(imp::Pixel16uC2& texel, float x, float y,
                                        float mul_x=1.0f, float mul_y=1.0f,
                                        float add_x=0.0f, float add_y=0.0f)
  {
    ushort2 val = this->fetch<ushort2>(x,y,mul_x,mul_y,add_x,add_y);
    texel = imp::Pixel16uC2(val.x, val.y);
  }
  //  __device__ __forceinline__ void fetch(imp::Pixel16uC3& texel, float x, float y, float mul_x=1.0f, float mul_y=1.0f, float add_x=0.0f, float add_y=0.0f)
  //  {
  //    ushort3 val = this->fetch<ushort3>(x,y,mul_x,mul_y,add_x,add_y);
  //    texel = imp::Pixel16uC3(val.x, val.y, val.z);
  //  }
  __device__ __forceinline__ void fetch(imp::Pixel16uC4& texel, float x, float y,
                                        float mul_x=1.0f, float mul_y=1.0f,
                                        float add_x=0.0f, float add_y=0.0f)
  {
    ushort4 val = this->fetch<ushort4>(x,y,mul_x,mul_y,add_x,add_y);
    texel = imp::Pixel16uC4(val.x, val.y, val.z, val.w);
  }

  __device__ __forceinline__ void fetch(imp::Pixel32sC1& texel, float x, float y,
                                        float mul_x=1.0f, float mul_y=1.0f,
                                        float add_x=0.0f, float add_y=0.0f)
  {
    texel = imp::Pixel32sC1(this->fetch<int1>(x,y,mul_x,mul_y,add_x,add_y).x);
  }
  __device__ __forceinline__ void fetch(imp::Pixel32sC2& texel, float x, float y,
                                        float mul_x=1.0f, float mul_y=1.0f,
                                        float add_x=0.0f, float add_y=0.0f)
  {
    int2 val = this->fetch<int2>(x,y,mul_x,mul_y,add_x,add_y);
    texel = imp::Pixel32sC2(val.x, val.y);
  }
  //  __device__ __forceinline__ void fetch(imp::Pixel32sC3& texel, float x, float y, float mul_x=1.0f, float mul_y=1.0f, float add_x=0.0f, float add_y=0.0f)
  //  {
  //    int3 val = this->fetch<int3>(x,y,mul_x,mul_y,add_x,add_y);
  //    texel = imp::Pixel32sC3(val.x, val.y, val.z);
  //  }
  __device__ __forceinline__ void fetch(imp::Pixel32sC4& texel, float x, float y,
                                        float mul_x=1.0f, float mul_y=1.0f,
                                        float add_x=0.0f, float add_y=0.0f)
  {
    int4 val = this->fetch<int4>(x,y,mul_x,mul_y,add_x,add_y);
    texel = imp::Pixel32sC4(val.x, val.y, val.z, val.w);
  }

  __device__ __forceinline__ void fetch(imp::Pixel32fC1& texel, float x, float y,
                                        float mul_x=1.0f, float mul_y=1.0f,
                                        float add_x=0.0f, float add_y=0.0f)
  {
    texel = imp::Pixel32fC1(this->fetch<float1>(x,y,mul_x,mul_y,add_x,add_y).x);
  }
  __device__ __forceinline__ void fetch(imp::Pixel32fC2& texel, float x, float y,
                                        float mul_x=1.0f, float mul_y=1.0f,
                                        float add_x=0.0f, float add_y=0.0f)
  {
    float2 val = this->fetch<float2>(x,y,mul_x,mul_y,add_x,add_y);
    texel = imp::Pixel32fC2(val.x, val.y);
  }
  //  __device__ __forceinline__ void fetch(imp::Pixel32fC3& texel, float x, float y, float mul_x=1.0f, float mul_y=1.0f, float add_x=0.0f, float add_y=0.0f)
  //  {
  //    float3 val = this->fetch<float3>(x,y,mul_x,mul_y,add_x,add_y);
  //    texel = imp::Pixel32fC3(val.x, val.y, val.z);
  //  }
  __device__ __forceinline__ void fetch(imp::Pixel32fC4& texel, float x, float y,
                                        float mul_x=1.0f, float mul_y=1.0f,
                                        float add_x=0.0f, float add_y=0.0f)
  {
    float4 val = this->fetch<float4>(x,y,mul_x,mul_y,add_x,add_y);
    texel = imp::Pixel32fC4(val.x, val.y, val.z, val.w);
  }

};


//------------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
Texture2D genTexture(const imp::cu::ImageGpu<Pixel, pixel_type>& img)
{
  cudaResourceDesc tex_res;
  std::memset(&tex_res, 0, sizeof(tex_res));
  tex_res.resType = cudaResourceTypePitch2D;
  tex_res.res.pitch2D.width = img.width();
  tex_res.res.pitch2D.height = img.height();
  tex_res.res.pitch2D.pitchInBytes = img.pitch();
  tex_res.res.pitch2D.devPtr = const_cast<void*>((const void*)img.cuData());
  tex_res.res.pitch2D.desc = img.channelFormatDesc();

  cudaTextureDesc tex_desc;
  std::memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.normalizedCoords = 0;
  tex_desc.filterMode = cudaFilterModeLinear;
  tex_desc.addressMode[0] = cudaAddressModeClamp;
  tex_desc.addressMode[1] = cudaAddressModeClamp;
  tex_desc.readMode = cudaReadModeElementType;

  cudaTextureObject_t tex_obj;
  cudaError_t err = cudaCreateTextureObject(&tex_obj, &tex_res, &tex_desc, 0);
  if  (err != ::cudaSuccess)
  {
    throw imp::cu::Exception("Failed to create texture object", err,
                             __FILE__, __FUNCTION__, __LINE__);
  }
  return Texture2D(tex_obj);
}


} // namespace cu
} // namespace imp

#endif // IMP_CU_TEXTURE_CUH


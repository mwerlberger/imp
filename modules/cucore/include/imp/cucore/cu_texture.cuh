#ifndef IMP_CU_TEXTURE_CUH
#define IMP_CU_TEXTURE_CUH

#include <cstring>
#include <cuda_runtime_api.h>
#include <imp/core/types.hpp>
#include <imp/core/pixel_enums.hpp>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cucore/cu_pixel_conversion.hpp>

namespace imp { namespace cu {


//template<typename Pixel>
struct Texture
{
  cudaTextureObject_t tex;
  bool normalized_coords;

  __host__ Texture() = default;
  __host__ virtual ~Texture() = default;
};

//template<typename Pixel, imp::PixelType pixel_type>
struct Texture2D : Texture
{
//  typedef Pixel pixel_t;
//  typedef pixel_t* pixel_container_t;

  std::uint32_t width, height;

  __host__ Texture2D(void* data, size_type pitch,
                     cudaChannelFormatDesc channel_desc,
                     imp::Size2u size,
                     bool _normalized_coords = false,
                     cudaTextureFilterMode filter_mode = cudaFilterModePoint,
                     cudaTextureAddressMode address_mode = cudaAddressModeClamp,
                     cudaTextureReadMode read_mode = cudaReadModeElementType)
    : Texture()
  {
    width = size.width();
    height = size.height();
    this->normalized_coords = _normalized_coords;

    cudaResourceDesc tex_res;
    std::memset(&tex_res, 0, sizeof(tex_res));
    tex_res.resType = cudaResourceTypePitch2D;
    tex_res.res.pitch2D.width = width;
    tex_res.res.pitch2D.height = height;
    tex_res.res.pitch2D.pitchInBytes = pitch;
    tex_res.res.pitch2D.devPtr = data;
    tex_res.res.pitch2D.desc = channel_desc;

    cudaTextureDesc tex_desc;
    std::memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.normalizedCoords = normalized_coords;
    tex_desc.filterMode = filter_mode;
    tex_desc.addressMode[0] = address_mode;
    tex_desc.addressMode[1] = address_mode;
    tex_desc.readMode = read_mode;

    cudaCreateTextureObject(&(this->tex), &tex_res, &tex_desc, 0);

  }

  __host__ ~Texture2D()
  {
    cudaDestroyTextureObject(this->tex);
  }

//  /**
//   * Wrapper for accesing texels. The coordinate are not texture but pixel coords (no need to add 0.5f!)
//   */
//  template<typename T=Pixel>
//  __device__ __forceinline__ T fetch(float x, float y) const
//  {
//    if (this->normalized_coords)
//    {
//      x /= width;
//      y /= height;
//    }
//    else
//    {
//      x += 0.5f;
//      y += 0.5f;
//    }
//    return tex2D<T>(tex, x, y);
//  }

};

//template<typename Pixel>
//template<typename T>
//__device__ T Texture2D<Pixel>::fetch(float x, float y) const
//{
//  if (this->normalized_coords)
//  {
//    x /= size.width();
//    y /= size.height();
//  }
//  else
//  {
//    x += 0.5f;
//    y += 0.5f;
//  }
//  return tex2D<float>(tex, x, y);
//}

} // namespace cu
} // namespace imp

#endif // IMP_CU_TEXTURE_CUH


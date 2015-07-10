#ifndef IMP_CU_IKC_DENOISING_CUH
#define IMP_CU_IKC_DENOISING_CUH

#include <memory>
#include <cstring>
#include <cuda_runtime_api.h>

#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/cu_core/cu_utils.hpp>

namespace imp {
namespace cu {

#if 0
//------------------------------------------------------------------------------
/**
 * @brief The Texture2D struct wrappes the cuda texture object
 */
struct Texture2D
{
  cudaTextureObject_t tex_object;
  __device__ __forceinline__ operator cudaTextureObject_t() const {return tex_object;}

  using Ptr = std::shared_ptr<Texture2D>;

  __host__ Texture2D()
    : tex_object(0)
  {
  }

  __host__ __device__ Texture2D(cudaTextureObject_t _tex_object)
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
};
//-----------------------------------------------------------------------------
template<typename T>
__device__ __forceinline__
T tex2DFetch(
    const Texture2D& tex, float x, float y,
    float mul_x=1.f, float mul_y=1.f, float add_x=0.f, float add_y=0.f)
{
  return ::tex2D<T>(tex.tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
}
#endif

//-----------------------------------------------------------------------------
class IterativeKernelCalls
{
public:
  IterativeKernelCalls();
  ~IterativeKernelCalls();

  void init(const Size2u& size);
  void run(const imp::cu::ImageGpu32fC1::Ptr& dst,
           const imp::cu::ImageGpu32fC1::Ptr& src,
           bool break_things=false);

  void breakThings();

private:
  imp::cu::ImageGpu32fC1::Ptr in_;
  imp::cu::ImageGpu32fC1::Ptr out_;
  std::unique_ptr<ImageGpu32fC1> unrelated_;

  // cuda textures
  std::shared_ptr<imp::cu::Texture2D> in_tex_;

  Size2u size_;
};

} // namespace cu
} // namespace imp

#endif // IMP_CU_IKC_DENOISING_CUH

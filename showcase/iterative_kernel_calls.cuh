#ifndef IMP_CU_IKC_DENOISING_CUH
#define IMP_CU_IKC_DENOISING_CUH

#include <sstream>

#include <memory>
#include <cstring>
#include <cuda_runtime_api.h>


namespace imp {
namespace cu {

//------------------------------------------------------------------------------
class Exception : public std::exception
{
public:
  Exception() = default;
  virtual ~Exception() throw() = default;

  Exception(const std::string& msg, cudaError err,
            const char* file=nullptr, const char* function=nullptr, int line=0) throw()
    : msg_(msg)
    , err_(err)
    , file_(file)
    , function_(function)
    , line_(line)
  {
    std::ostringstream out_msg;

    out_msg << "IMP Exception (CUDA): ";
    out_msg << (msg_.empty() ? "unknown error" : msg_) << "\n";
    out_msg << "      cudaError code: " << cudaGetErrorString(err_);
    out_msg << " (" << err_ << ")" << "\n";
    out_msg << "      where: ";
    out_msg << (file_.empty() ? "no filename available" : file_) << " | ";
    out_msg << (function_.empty() ? "unknown function" : function_) << ":" << line_;
    msg_ = out_msg.str();
  }

  virtual const char* what() const throw()
  {
    return msg_.c_str();
  }

  std::string msg_;
  cudaError err_;
  std::string file_;
  std::string function_;
  int line_;
};


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

  __host__ Texture2D(const void* data, size_t pitch,
                     cudaChannelFormatDesc channel_desc,
                     std::uint32_t width, std::uint32_t height,
                     bool _normalized_coords = false,
                     cudaTextureFilterMode filter_mode = cudaFilterModePoint,
                     cudaTextureAddressMode address_mode = cudaAddressModeClamp,
                     cudaTextureReadMode read_mode = cudaReadModeElementType)
  {
    cudaResourceDesc tex_res;
    std::memset(&tex_res, 0, sizeof(tex_res));
    tex_res.resType = cudaResourceTypePitch2D;
    tex_res.res.pitch2D.width = width;
    tex_res.res.pitch2D.height = height;
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
      throw Exception("Failed to create texture object", err,
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
class IterativeKernelCalls
{
public:
  IterativeKernelCalls();
  ~IterativeKernelCalls();

  void run(float* dst, const float* src, size_t pitch,
           std::uint32_t width, std::uint32_t height,
           bool break_things=false);

  void breakThings();

private:
  void init();

  const float* in_buffer_;
  float* out_buffer_;
  size_t pitch_;
  float* unrelated_;
  size_t unrelated_pitch_;

  // cuda textures
  std::shared_ptr<imp::cu::Texture2D> in_tex_;

  std::uint32_t width_;
  std::uint32_t height_;
};

} // namespace cu
} // namespace imp

#endif // IMP_CU_IKC_DENOISING_CUH

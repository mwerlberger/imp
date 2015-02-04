#ifndef IMP_CU_VARIATIONAL_DENOISING_CUH
#define IMP_CU_VARIATIONAL_DENOISING_CUH

#include <memory>
#include <cuda_runtime_api.h>
#include <imp/cucore/cu_image_gpu.cuh>

namespace imp { namespace cu {

template<typename Pixel, imp::PixelType pixel_type>
class VariationalDenoising
{
public:
  typedef imp::cu::ImageGpu<Pixel, pixel_type> Image;
  typedef std::shared_ptr<Image> ImagePtr;

public:
  VariationalDenoising() = default;
  virtual ~VariationalDenoising() = default;

  virtual __host__ void denoise(ImagePtr f, ImagePtr u) = 0;

protected:
  ImagePtr f_;
  ImagePtr u_;

  std::shared_ptr<imp::ImageBase> u_prev_;
  std::shared_ptr<imp::ImageBase> p_;

  // cuda textures
  cudaTextureObject_t f_tex_;
  cudaTextureObject_t u_tex_;
  cudaTextureObject_t u_prev_tex_;
  cudaTextureObject_t p_tex_;

  Size2u size_;

  // algorithm parameters
  struct Parameters
  {
    float lambda_ = 1.0f;
    std::uint16_t max_iter_ = 100;
  };
  Parameters params_;

};

} // namespace cu
} // namespace imp

#endif // IMP_CU_VARIATIONAL_DENOISING_CUH


#ifndef IMP_CU_VARIATIONAL_DENOISING_CUH
#define IMP_CU_VARIATIONAL_DENOISING_CUH

#include <memory>
#include <cuda_runtime_api.h>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cucore/cu_texture.cuh>

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
  std::shared_ptr<imp::cu::ImageGpu32fC1> u_;

  std::shared_ptr<imp::cu::ImageGpu32fC1> u_prev_;
  std::shared_ptr<imp::cu::ImageGpu32fC2> p_;

  // cuda textures
  std::shared_ptr<imp::cu::Texture2D> f_tex_;
  std::shared_ptr<imp::cu::Texture2D> u_tex_;
  std::shared_ptr<imp::cu::Texture2D> u_prev_tex_;
  std::shared_ptr<imp::cu::Texture2D> p_tex_;

  Size2u size_;

  // algorithm parameters
  struct Parameters
  {
    float lambda = 1.0f;
    std::uint16_t max_iter = 100;
  };
  Parameters params_;

};

} // namespace cu
} // namespace imp

#endif // IMP_CU_VARIATIONAL_DENOISING_CUH


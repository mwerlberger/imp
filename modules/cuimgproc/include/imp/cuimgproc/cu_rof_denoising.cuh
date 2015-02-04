#ifndef IMP_CU_ROF_DENOISING_CUH
#define IMP_CU_ROF_DENOISING_CUH

#include <imp/cuimgproc/cu_variational_denoising.cuh>
#include <cuda_runtime_api.h>

namespace imp { namespace cu {

template<typename Pixel, imp::PixelType pixel_type>
class RofDenoising  : public imp::cu::VariationalDenoising<Pixel, pixel_type>
{
public:
  typedef VariationalDenoising<Pixel, pixel_type> Base;
  typedef imp::cu::ImageGpu<Pixel, pixel_type> Image;
  typedef std::shared_ptr<Image> ImagePtr;


public:
  RofDenoising() = default;
  virtual ~RofDenoising() = default;
  using Base::VariationalDenoising;

  virtual __host__ void denoise(ImagePtr f, ImagePtr u) override;

protected:
};

//-----------------------------------------------------------------------------
// convenience typedefs
// (sync with explicit template class instantiations at the end of the cpp file)
typedef RofDenoising<imp::Pixel8uC1, imp::PixelType::i8uC1> RofDenoising8uC1;
typedef RofDenoising<imp::Pixel32fC1, imp::PixelType::i32fC1> RofDenoising32fC1;

} // namespace cu
} // namespace imp

#endif // IMP_CU_ROF_DENOISING_CUH

#ifndef IMP_CU_ROF_DENOISING_CUH
#define IMP_CU_ROF_DENOISING_CUH

#include <imp/cuimgproc/cu_variational_denoising.cuh>

#include <memory>
#include <cuda_runtime_api.h>

#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cucore/cu_utils.hpp>

namespace imp { namespace cu {

template<typename Pixel, imp::PixelType pixel_type>
class RofDenoising  : public imp::cu::VariationalDenoising
{
public:
  typedef VariationalDenoising Base;
  typedef imp::cu::ImageGpu<Pixel, pixel_type> Image;
  typedef std::shared_ptr<Image> ImagePtr;

public:
  RofDenoising() = default;
  virtual ~RofDenoising() = default;
  using Base::VariationalDenoising;

  virtual __host__ void init(const Size2u& size) override;
  virtual __host__ void denoise(const std::shared_ptr<imp::ImageBase>& dst,
                                const std::shared_ptr<imp::ImageBase>& src) override;

protected:
  virtual void print(std::ostream &os) const override;

private:
  ImagePtr f_;

};

//-----------------------------------------------------------------------------
// convenience typedefs
// (sync with explicit template class instantiations at the end of the cpp file)
typedef RofDenoising<imp::Pixel8uC1, imp::PixelType::i8uC1> RofDenoising8uC1;
typedef RofDenoising<imp::Pixel32fC1, imp::PixelType::i32fC1> RofDenoising32fC1;

} // namespace cu
} // namespace imp

#endif // IMP_CU_ROF_DENOISING_CUH

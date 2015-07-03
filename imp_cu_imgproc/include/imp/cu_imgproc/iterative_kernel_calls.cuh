#ifndef IMP_CU_IKC_DENOISING_CUH
#define IMP_CU_IKC_DENOISING_CUH

#include <memory>
#include <cuda_runtime_api.h>

#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/cu_core/cu_utils.hpp>
#include <imp/cu_imgproc/cu_variational_denoising.cuh>

namespace imp {
namespace cu {

class IterativeKernelCalls  : public imp::cu::VariationalDenoising
{
public:
  using Base = VariationalDenoising;
  using ImageGpu = imp::cu::ImageGpu32fC1;
  using Ptr = std::shared_ptr<IterativeKernelCalls>;

public:
  IterativeKernelCalls() = default;
  virtual ~IterativeKernelCalls() = default;
  using Base::Base;

  virtual void init(const Size2u& size) override;
  virtual void denoise(const std::shared_ptr<imp::ImageBase>& dst,
                       const std::shared_ptr<imp::ImageBase>& src) override;

  void breakThings();

protected:
  virtual void print(std::ostream &os) const override;

private:
  typename ImageGpu::Ptr f_;

  // pixel-wise primal and dual energies to avoid allocation of memory for every check
  std::unique_ptr<ImageGpu32fC1> primal_energies_;
  std::unique_ptr<ImageGpu32fC1> dual_energies_;

};

} // namespace cu
} // namespace imp

#endif // IMP_CU_IKC_DENOISING_CUH

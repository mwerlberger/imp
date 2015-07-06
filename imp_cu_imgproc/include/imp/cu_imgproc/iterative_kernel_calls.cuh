#ifndef IMP_CU_IKC_DENOISING_CUH
#define IMP_CU_IKC_DENOISING_CUH

#include <memory>
#include <cuda_runtime_api.h>

#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/cu_core/cu_utils.hpp>

namespace imp {
namespace cu {

// forward declarations
class Texture2D;

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

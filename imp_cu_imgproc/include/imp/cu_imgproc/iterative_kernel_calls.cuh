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

struct ikcParams
{
  float lambda = 10.f;
  std::uint16_t max_iter = 100;
  bool verbose = false;
  std::uint16_t primal_dual_energy_check_iter = 0;
  double primal_dual_gap_tolerance = 0.0;
};

class IterativeKernelCalls
{
public:
  using ImageGpu = imp::cu::ImageGpu32fC1;
  using Ptr = std::shared_ptr<IterativeKernelCalls>;
  using Fragmentation = imp::cu::Fragmentation<16>;
  using FragmentationPtr = std::shared_ptr<Fragmentation>;

public:
  IterativeKernelCalls();
  ~IterativeKernelCalls();

  virtual void init(const Size2u& size);
  virtual void denoise(const std::shared_ptr<imp::ImageBase>& dst,
                       const std::shared_ptr<imp::ImageBase>& src,
                       bool break_things=false);

  void breakThings();

  inline dim3 dimGrid() {return fragmentation_->dimGrid;}
  inline dim3 dimBlock() {return fragmentation_->dimBlock;}
  virtual inline ikcParams& params() { return params_; }

  friend std::ostream& operator<<(std::ostream& os,
                                  const IterativeKernelCalls& rhs);

protected:

  inline virtual void print(std::ostream& os) const
  {
    os << "IterativeKernelCalls parametrization:" << std::endl
       << "  lambda: " << this->params_.lambda << std::endl
       << "  max_iter: " << this->params_.max_iter << std::endl
       << "  primal_dual_energy_check_iter: " << this->params_.primal_dual_energy_check_iter << std::endl
       << "  primal_dual_gap_tolerance: " << this->params_.primal_dual_gap_tolerance << std::endl
       << std::endl;
  }

private:
  imp::cu::ImageGpu32fC1::Ptr f_;
  imp::cu::ImageGpu32fC1::Ptr u_;
  imp::cu::ImageGpu32fC1::Ptr u_prev_;
  imp::cu::ImageGpu32fC2::Ptr p_;

  // cuda textures
  std::shared_ptr<imp::cu::Texture2D> f_tex_;
  std::shared_ptr<imp::cu::Texture2D> u_tex_;
  std::shared_ptr<imp::cu::Texture2D> u_prev_tex_;
  std::shared_ptr<imp::cu::Texture2D> p_tex_;

  Size2u size_;
  FragmentationPtr fragmentation_;

  // algorithm parameters
  ikcParams params_;


  // pixel-wise primal and dual energies to avoid allocation of memory for every check
  std::unique_ptr<ImageGpu32fC1> unrelated_;
  std::unique_ptr<ImageGpu32fC1> dual_energies_;

};

inline std::ostream& operator<<(std::ostream& os,
                                const IterativeKernelCalls& rhs)
{
  rhs.print(os);
  return os;
}

} // namespace cu
} // namespace imp

#endif // IMP_CU_IKC_DENOISING_CUH

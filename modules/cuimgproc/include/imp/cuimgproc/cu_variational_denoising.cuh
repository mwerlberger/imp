#ifndef IMP_CU_VARIATIONAL_DENOISING_CUH
#define IMP_CU_VARIATIONAL_DENOISING_CUH

#include <memory>
#include <cuda_runtime_api.h>
#include <imp/core/image_base.hpp>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cucore/cu_utils.hpp>

namespace imp { namespace cu {

// forward declarations
class Texture2D;

/**
 * @brief The VariationalDenoising class
 */
class VariationalDenoising
{
public:
  typedef imp::cu::Fragmentation<16> CuFrag;
  typedef std::shared_ptr<CuFrag> CuFragPtr;

public:
  VariationalDenoising() = default;
  virtual ~VariationalDenoising() = default;

  inline virtual __host__ void init(Size2u size)
  {
    size_ = size;
    fragmentation_.reset(new CuFrag(size));

    // setup internal memory
    this->u_.reset(new ImageGpu32fC1(size));
    this->u_prev_.reset(new ImageGpu32fC1(size));
    this->p_.reset(new ImageGpu32fC2(size));
  }

  virtual __host__ void denoise(std::shared_ptr<imp::ImageBase> dst,
                                std::shared_ptr<imp::ImageBase> src) = 0;

  inline __host__ __device__ dim3 dimGrid() {return fragmentation_->dimGrid;}
  inline __host__ __device__ dim3 dimBlock() {return fragmentation_->dimBlock;}

  friend std::ostream& operator<<(std::ostream& os,
                                  const VariationalDenoising& rhs);

protected:

  inline virtual void print(std::ostream& os) const
  {
    //os << "  size: " << this->size_ << std::endl
    os << "  lambda: " << this->params_.lambda << std::endl
       << "  max_iter: " << this->params_.max_iter << std::endl;
  }


  std::shared_ptr<imp::cu::ImageGpu32fC1> u_;
  std::shared_ptr<imp::cu::ImageGpu32fC1> u_prev_;
  std::shared_ptr<imp::cu::ImageGpu32fC2> p_;

  // cuda textures
  std::shared_ptr<imp::cu::Texture2D> f_tex_;
  std::shared_ptr<imp::cu::Texture2D> u_tex_;
  std::shared_ptr<imp::cu::Texture2D> u_prev_tex_;
  std::shared_ptr<imp::cu::Texture2D> p_tex_;

  Size2u size_;
  CuFragPtr fragmentation_;

  // algorithm parameters
  struct Parameters
  {
    float lambda = 10.f;
    std::uint16_t max_iter = 100;
    bool verbose = false;
  };
  Parameters params_;

};

inline std::ostream& operator<<(std::ostream& os,
                                const VariationalDenoising& rhs)
{
  rhs.print(os);
  return os;
}


} // namespace cu
} // namespace imp

#endif // IMP_CU_VARIATIONAL_DENOISING_CUH


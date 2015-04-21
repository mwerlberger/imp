#ifndef IMP_CU_SOLVER_STEREO_ABSTRACT_HPP
#define IMP_CU_SOLVER_STEREO_ABSTRACT_HPP

#include <cstdint>
#include <memory>

#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/core/size.hpp>


namespace imp {
namespace cu {

// forward decl
class VariationalStereoParameters;

/**
 * @brief The StereoCtFWarpingLevel class
 */
class SolverStereoAbstract
{
public:
  using Parameters = VariationalStereoParameters;

  using Image = imp::cu::ImageGpu32fC1;
  using ImagePtr = std::shared_ptr<Image>;

  using DisparityImage = imp::cu::ImageGpu32fC1;
  using DisparityImagePtr = std::shared_ptr<Image>;

  using ConstImagePtrRef = const std::shared_ptr<Image>&;

  using VectorImage = imp::cu::ImageGpu32fC2;
  using VectorImagePtr = std::shared_ptr<VectorImage>;
  using ConstVectorImagePtr = const std::shared_ptr<VectorImage>&;


public:
  SolverStereoAbstract() = delete;
  virtual ~SolverStereoAbstract() = default;

  SolverStereoAbstract(std::shared_ptr<Parameters> params,
                        imp::Size2u size, std::uint16_t level)
    : params_(params)
    , size_(size)
    , level_(level)
  { ; }

  virtual void init() = 0;
  virtual void init(const SolverStereoAbstract& rhs) = 0;
  virtual void solve(std::vector<ImagePtr> images) = 0;
  virtual ImagePtr getDisparities() = 0;

  // setters / getters
  inline imp::Size2u size() { return size_; }
  inline std::uint16_t level() { return level_; }

protected:
  std::shared_ptr<Parameters> params_; //!< configuration parameters
  imp::Size2u size_;
  std::uint16_t level_; //!< level number in the ctf pyramid (0=finest .. n=coarsest)
};

} // namespace cu
} // namespace imp

#endif // IMP_CU_SOLVER_STEREO_ABSTRACT_HPP

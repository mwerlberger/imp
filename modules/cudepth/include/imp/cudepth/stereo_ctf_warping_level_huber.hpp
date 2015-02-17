#ifndef IMP_CU_STEREO_CTF_WARPING_LEVEL_HUBER_HPP
#define IMP_CU_STEREO_CTF_WARPING_LEVEL_HUBER_HPP

#include <cstdint>

#include <imp/cudepth/stereo_ctf_warping_level.hpp>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/core/size.hpp>


namespace imp {
namespace cu {

// forward decl
class VariationalStereoParameters;


/**
 * @brief The StereoCtFWarpingLevelHuber class
 */
class StereoCtFWarpingLevelHuber : public StereoCtFWarpingLevel
{
public:
  using Parameters = VariationalStereoParameters;
  using Image = imp::cu::ImageGpu32fC1;
  using Dual = imp::cu::ImageGpu32fC2;
  using ImagePtr = std::shared_ptr<Image>;



public:
  StereoCtFWarpingLevelHuber() = delete;
  virtual ~StereoCtFWarpingLevelHuber();

  StereoCtFWarpingLevelHuber(std::shared_ptr<Parameters> params,
                        imp::Size2u size, std::uint16_t level);

  virtual void init();
  virtual void init(const StereoCtFWarpingLevelHuber& from);
  virtual void solve(std::vector<ImagePtr> images);

protected:
  std::unique_ptr<Image> u_; //!< disparities (result)
  std::unique_ptr<Image> u_prev_; //!< disparities results from previous iteration
  std::unique_ptr<Image> u0_; //!< disparities results from previous warp
  std::unique_ptr<Dual> pu_; //!< dual variable for primal variable
  std::unique_ptr<Image> q_; //!< dual variable for data term
  std::unique_ptr<Image> ix_; //!< spatial gradients on moving (warped) image
  std::unique_ptr<Image> it_; //!< temporal gradients between warped and fixed image


  std::shared_ptr<Parameters> params_; //!< configuration parameters
  imp::Size2u size_;
  std::uint16_t level_; //!< level number in the ctf pyramid (0=finest .. n=coarsest)
};

} // namespace cu
} // namespace imp

#endif // IMP_CU_STEREO_CTF_WARPING_LEVEL_HUBER_HPP

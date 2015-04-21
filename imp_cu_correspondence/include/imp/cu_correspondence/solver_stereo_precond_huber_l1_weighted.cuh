#ifndef IMP_CU_STEREO_CTF_WARPING_LEVEL_PRECOND_HUBER_L1_WEIGHTED_CUH
#define IMP_CU_STEREO_CTF_WARPING_LEVEL_PRECOND_HUBER_L1_WEIGHTED_CUH

#include <cstdint>
#include <memory>

#include <imp/cu_correspondence/solver_stereo_abstract.hpp>
#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/core/size.hpp>

namespace imp {
namespace cu {

// forward decl
class VariationalStereoParameters;
class Texture2D;

/**
 * @brief The StereoCtFWarpingLevelPrecondHuberL1 class
 * @todo better polymorphism! (derive from StereoCtfWarpingLevelPrecondHuberL1
 */
class StereoCtFWarpingLevelPrecondHuberL1Weighted : public SolverStereoAbstract
{
public:
  using Parameters = VariationalStereoParameters;
  using Image = imp::cu::ImageGpu32fC1;
  using Dual = imp::cu::ImageGpu32fC2;
  using ImagePtr = std::shared_ptr<Image>;


public:
  StereoCtFWarpingLevelPrecondHuberL1Weighted() = delete;
  virtual ~StereoCtFWarpingLevelPrecondHuberL1Weighted();

  StereoCtFWarpingLevelPrecondHuberL1Weighted(const std::shared_ptr<Parameters>& params,
                                              imp::Size2u size, size_type level);

  virtual void init() override;
  virtual void init(const SolverStereoAbstract& rhs) override;
  virtual void solve(std::vector<ImagePtr> images) override;

  virtual inline ImagePtr getDisparities() override {return u_;}


protected:
  ImagePtr u_; //!< disparities (result)
  std::unique_ptr<Image> u_prev_; //!< disparities results from previous iteration
  std::unique_ptr<Image> u0_; //!< disparities results from previous warp
  std::unique_ptr<Dual> pu_; //!< dual variable for primal variable
  std::unique_ptr<Image> q_; //!< dual variable for data term
  std::unique_ptr<Image> ix_; //!< spatial gradients on moving (warped) image
  std::unique_ptr<Image> it_; //!< temporal gradients between warped and fixed image
  std::unique_ptr<Image> xi_; //!< preconditioner
  std::unique_ptr<Image> g_; //!< (edge) image for weighting the regularizer

  // textures
  std::unique_ptr<Texture2D> lambda_tex_; //!< For pointwise lambda
  std::unique_ptr<Texture2D> i1_tex_;
  std::unique_ptr<Texture2D> i2_tex_;
  std::unique_ptr<Texture2D> u_tex_;
  std::unique_ptr<Texture2D> u_prev_tex_;
  std::unique_ptr<Texture2D> u0_tex_;
  std::unique_ptr<Texture2D> pu_tex_;
  std::unique_ptr<Texture2D> q_tex_;
  std::unique_ptr<Texture2D> ix_tex_;
  std::unique_ptr<Texture2D> it_tex_;
  std::unique_ptr<Texture2D> xi_tex_;
  std::unique_ptr<Texture2D> g_tex_;

};

} // namespace cu
} // namespace imp

#endif // IMP_CU_STEREO_CTF_WARPING_LEVEL_PRECOND_HUBER_L1_WEIGHTED_CUH

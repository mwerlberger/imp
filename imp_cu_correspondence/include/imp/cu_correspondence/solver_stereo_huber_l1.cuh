#pragma once

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
 * @brief The SolverStereoHuberL1 class computes the disparities between two views
 *        by using a Huber-L1 Regularization-Dataterm combination
 *        optimized with a primal-dual optimization.
 */
class SolverStereoHuberL1 : public SolverStereoAbstract
{
public:
  using Parameters = VariationalStereoParameters;
  using Image = imp::cu::ImageGpu32fC1;
  using Dual = imp::cu::ImageGpu32fC2;
  using ImagePtr = std::shared_ptr<Image>;


public:
  SolverStereoHuberL1() = delete;
  virtual ~SolverStereoHuberL1();

  SolverStereoHuberL1(const std::shared_ptr<Parameters>& params,
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
  std::unique_ptr<Image> ix_; //!< spatial gradients on moving (warped) image
  std::unique_ptr<Image> it_; //!< temporal gradients between warped and fixed image

  // textures
  std::unique_ptr<Texture2D> i1_tex_;
  std::unique_ptr<Texture2D> i2_tex_;
  std::unique_ptr<Texture2D> u_tex_;
  std::unique_ptr<Texture2D> u_prev_tex_;
  std::unique_ptr<Texture2D> u0_tex_;
  std::unique_ptr<Texture2D> pu_tex_;
  std::unique_ptr<Texture2D> ix_tex_;
  std::unique_ptr<Texture2D> it_tex_;

};

} // namespace cu
} // namespace imp

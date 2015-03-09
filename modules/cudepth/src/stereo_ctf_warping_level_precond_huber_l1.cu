#include <imp/cudepth/stereo_ctf_warping_level_precond_huber_l1.cuh>

#include <cuda_runtime.h>

#include <imp/cudepth/variational_stereo_parameters.hpp>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cuimgproc/cu_image_filter.cuh>
#include <imp/cuimgproc/cu_image_transform.cuh>
#include <imp/cucore/cu_utils.hpp>
#include <imp/cucore/cu_texture.cuh>
#include <imp/cucore/cu_math.cuh>

#include <imp/io/opencv_bridge.hpp>

#include "cu_k_warped_gradients.cuh"
#include "cu_k_stereo_ctf_warping_level_huber.cuh"

namespace imp {
namespace cu {

//------------------------------------------------------------------------------
StereoCtFWarpingLevelPrecondHuberL1::~StereoCtFWarpingLevelPrecondHuberL1()
{
  // thanks to smart pointers
}

//------------------------------------------------------------------------------
StereoCtFWarpingLevelPrecondHuberL1::StereoCtFWarpingLevelPrecondHuberL1(
    const std::shared_ptr<Parameters>& params, imp::Size2u size, size_type level)
  : StereoCtFWarpingLevel(params, size, level)
{
  u_.reset(new Image(size));
  u_prev_.reset(new Image(size));
  u0_.reset(new Image(size));
  pu_.reset(new Dual(size));
  q_.reset(new Image(size));
  ix_.reset(new Image(size));
  it_.reset(new Image(size));
  xi_.reset(new Image(size));

  // and its textures
  u_tex_ = u_->genTexture(false, cudaFilterModeLinear);
  u_prev_tex_ =  u_prev_->genTexture(false, cudaFilterModeLinear);
  u0_tex_ =  u0_->genTexture(false, cudaFilterModeLinear);
  pu_tex_ =  pu_->genTexture(false, cudaFilterModeLinear);
  q_tex_ =  q_->genTexture(false, cudaFilterModeLinear);
  ix_tex_ =  ix_->genTexture(false, cudaFilterModeLinear);
  it_tex_ =  it_->genTexture(false, cudaFilterModeLinear);
  xi_tex_ =  xi_->genTexture(false, cudaFilterModeLinear);
}

//------------------------------------------------------------------------------
void StereoCtFWarpingLevelPrecondHuberL1::init()
{
  u_->setValue(0.0f);
  u_prev_->setValue(0.0f);
  u0_->setValue(0.0f);
  pu_->setValue(0.0f);
  q_->setValue(0.0f);
}

//------------------------------------------------------------------------------
void StereoCtFWarpingLevelPrecondHuberL1::init(const StereoCtFWarpingLevel& rhs)
{
  const StereoCtFWarpingLevelPrecondHuberL1* from =
      dynamic_cast<const StereoCtFWarpingLevelPrecondHuberL1*>(&rhs);

  float inv_sf = 1./params_->ctf.scale_factor; // >1 for adapting prolongated disparities

  std::cout << "inv_sf: " << inv_sf << std::endl;
  {
    imp::Pixel32fC1 min_val,max_val;
    imp::cu::minMax(from->u_, min_val, max_val);
    std::cout << "disp: min: " << min_val.x << " max: " << max_val.x << std::endl;
  }

  if(params_->ctf.apply_median_filter)
  {
    imp::cu::filterMedian3x3(from->u0_.get(), from->u_.get());
    imp::cu::resample(u_.get(), from->u0_.get(), imp::InterpolationMode::linear, false);
  }
  else
  {
    imp::cu::resample(u_.get(), from->u_.get(), imp::InterpolationMode::linear, false);
  }
  *u_ *= inv_sf;
  std::cout << "inv_sf: " << inv_sf << std::endl;
  {
    imp::Pixel32fC1 min_val,max_val;
    imp::cu::minMax(u_, min_val, max_val);
    std::cout << "disp: min: " << min_val.x << " max: " << max_val.x << std::endl;
  }

  imp::cu::resample(pu_.get(), from->pu_.get(), imp::InterpolationMode::linear, false);
  imp::cu::resample(q_.get(), from->q_.get(), imp::InterpolationMode::linear, false);
}

//------------------------------------------------------------------------------
void StereoCtFWarpingLevelPrecondHuberL1::solve(std::vector<ImagePtr> images)
{
  std::cout << "StereoCtFWarpingLevelPrecondHuberL1: solving level " << level_ << " with " << images.size() << " images" << std::endl;

  // sanity check:
  // TODO

  // image textures
  i1_tex_ = images.at(0)->genTexture(false, cudaFilterModeLinear);
  i2_tex_ = images.at(1)->genTexture(false, cudaFilterModeLinear);

  Fragmentation<16,16> frag(size_);


  // constants
  constexpr float tau = 0.95f;
  constexpr float sigma = 0.95f;
  float lin_step = 0.5f;

  // precond
  constexpr float eta = 2.0f;

  // warping
  for (std::uint32_t warp = 0; warp < params_->ctf.warps; ++warp)
  {
    std::cout << "warp" << std::endl;
    u_->copyTo(*u0_);

    // compute warped spatial and temporal gradients
    k_warpedGradients
        <<<
          frag.dimGrid, frag.dimBlock
        >>> (ix_->data(), it_->data(), ix_->stride(), ix_->width(), ix_->height(),
             *i1_tex_, *i2_tex_, *u0_tex_);

    // compute preconditioner
    k_preconditioner
        <<<
          frag.dimGrid, frag.dimBlock
        >>> (xi_->data(), xi_->stride(), xi_->width(), xi_->height(),
             params_->lambda, *ix_tex_);


    for (std::uint32_t iter = 0; iter < params_->ctf.iters; ++iter)
    {
      // dual kernel
      k_dualUpdate
          <<<
            frag.dimGrid, frag.dimBlock
          >>> (pu_->data(), pu_->stride(), q_->data(), q_->stride(),
               size_.width(), size_.height(),
               params_->lambda, params_->eps_u, sigma, eta,
               *u_prev_tex_, *u0_tex_, *pu_tex_, *q_tex_, *ix_tex_, *it_tex_);

      // and primal kernel
      k_primalUpdate
          <<<
            frag.dimGrid, frag.dimBlock
          >>> (u_->data(), u_prev_->data(), u_->stride(),
               size_.width(), size_.height(),
               params_->lambda, tau, lin_step,
               *u_tex_, *u0_tex_, *pu_tex_, *q_tex_, *ix_tex_, *xi_tex_);

      if (iter % 50)
      {
        imp::cu::ocvBridgeShow("current disp", *u_, true);
        imp::cu::ocvBridgeShow("current i0", *images.at(0), true);
        cv::waitKey(1);
      }

    } // iters
    lin_step /= 1.2f;

  } // warps
  IMP_CUDA_CHECK();
}



} // namespace cu
} // namespace imp


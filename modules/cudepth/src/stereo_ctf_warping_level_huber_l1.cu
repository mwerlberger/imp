#include <imp/cudepth/stereo_ctf_warping_level_huber_l1.cuh>

#include <cmath>

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
#include "cu_k_stereo_ctf_warping_level_huber_l1.cuh"

namespace imp {
namespace cu {


//------------------------------------------------------------------------------
StereoCtFWarpingLevelHuberL1::~StereoCtFWarpingLevelHuberL1()
{
  // thanks to smart pointers
}

//------------------------------------------------------------------------------
StereoCtFWarpingLevelHuberL1::StereoCtFWarpingLevelHuberL1(
    const std::shared_ptr<Parameters>& params, imp::Size2u size, size_type level)
  : StereoCtFWarpingLevel(params, size, level)
{
  u_.reset(new Image(size));
  u_prev_.reset(new Image(size));
  u0_.reset(new Image(size));
  pu_.reset(new Dual(size));
  ix_.reset(new Image(size));
  it_.reset(new Image(size));

  // and its textures
  u_tex_ = u_->genTexture(false, cudaFilterModeLinear);
  u_prev_tex_ =  u_prev_->genTexture(false, cudaFilterModeLinear);
  u0_tex_ =  u0_->genTexture(false, cudaFilterModeLinear);
  pu_tex_ =  pu_->genTexture(false, cudaFilterModeLinear);
  ix_tex_ =  ix_->genTexture(false, cudaFilterModeLinear);
  it_tex_ =  it_->genTexture(false, cudaFilterModeLinear);
}

//------------------------------------------------------------------------------
void StereoCtFWarpingLevelHuberL1::init()
{
  u_->setValue(0.0f);
  pu_->setValue(0.0f);
  // other variables are init and/or set when needed!
}

//------------------------------------------------------------------------------
void StereoCtFWarpingLevelHuberL1::init(const StereoCtFWarpingLevel& rhs)
{
  const StereoCtFWarpingLevelHuberL1* from =
      dynamic_cast<const StereoCtFWarpingLevelHuberL1*>(&rhs);

  float inv_sf = 1./params_->ctf.scale_factor; // >1 for adapting prolongated disparities

  if(params_->ctf.apply_median_filter)
  {
    imp::cu::filterMedian3x3(from->u0_.get(), from->u_.get());
    imp::cu::resample(u_.get(), from->u0_.get(), imp::InterpolationMode::point, false);
  }
  else
  {
    imp::cu::resample(u_.get(), from->u_.get(), imp::InterpolationMode::point, false);
  }
  *u_ *= inv_sf;

  imp::cu::resample(pu_.get(), from->pu_.get(), imp::InterpolationMode::point, false);
}

//------------------------------------------------------------------------------
void StereoCtFWarpingLevelHuberL1::solve(std::vector<ImagePtr> images)
{
  std::cout << "StereoCtFWarpingLevelPrecondHuberL1: solving level " << level_ << " with " << images.size() << " images" << std::endl;

  // sanity check:
  // TODO

  i1_tex_ = images.at(0)->genTexture(false, cudaFilterModeLinear);
  i2_tex_ = images.at(1)->genTexture(false, cudaFilterModeLinear);
  u_->copyTo(*u_prev_);
  Fragmentation<16,16> frag(size_);

  // constants
  const float L = std::sqrt(8.f);
  const float tau = 1.f/L;
  const float sigma = 1.f/L;
  float lin_step = 0.5f;

  // warping
  for (std::uint32_t warp = 0; warp < params_->ctf.warps; ++warp)
  {
    if (params_->verbose > 5)
      std::cout << "SOLVING warp iteration of Huber-L1 stereo model." << std::endl;

    if (false && params_->ctf.apply_median_filter)
    {
      imp::cu::filterMedian3x3(u0_.get(), u_.get());
    }
    else
    {
      u_->copyTo(*u0_);
    }

    // compute warped spatial and temporal gradients
    k_warpedGradients
        <<<
          frag.dimGrid, frag.dimBlock
        >>> (ix_->data(), it_->data(), ix_->stride(), ix_->width(), ix_->height(),
             *i1_tex_, *i2_tex_, *u0_tex_);

    if (params_->verbose > 10)
    {
      imp::cu::ocvBridgeShow("ix", *ix_, true);
      imp::cu::ocvBridgeShow("it", *it_, true);
    }

    for (std::uint32_t iter = 0; iter < params_->ctf.iters; ++iter)
    {
      // dual kernel
      k_dualUpdate
          <<<
            frag.dimGrid, frag.dimBlock
          >>> (pu_->data(), pu_->stride(),
               size_.width(), size_.height(),
               params_->eps_u, sigma,
               *u_prev_tex_, *pu_tex_);

      // and primal kernel
      k_primalUpdate
          <<<
            frag.dimGrid, frag.dimBlock
          >>> (u_->data(), u_prev_->data(), u_->stride(),
               size_.width(), size_.height(),
               params_->lambda, tau, lin_step,
               *u_tex_, *u0_tex_, *pu_tex_, *ix_tex_, *it_tex_);

      if (params_->verbose > 5 && iter % 50)
      {
        imp::cu::ocvBridgeShow("current disp", *u_, true);
        imp::cu::ocvBridgeShow("current i0", *images.at(0), true);
        cv::waitKey(1);
      }

    } // iters
//    lin_step /= 1.2f;

  } // warps
  IMP_CUDA_CHECK();
}



} // namespace cu
} // namespace imp


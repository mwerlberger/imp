#include <imp/cudepth/stereo_ctf_warping_level_huber.cuh>

#include <cuda_runtime.h>

#include <imp/cudepth/variational_stereo_parameters.hpp>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cuimgproc/cu_image_filter.cuh>
#include <imp/cuimgproc/cu_image_transform.cuh>
#include <imp/cucore/cu_utils.hpp>
#include <imp/cucore/cu_texture.cuh>

namespace imp {
namespace cu {

//------------------------------------------------------------------------------
StereoCtFWarpingLevelHuber::~StereoCtFWarpingLevelHuber()
{
  // thanks to smart pointers
}

//------------------------------------------------------------------------------
StereoCtFWarpingLevelHuber::StereoCtFWarpingLevelHuber(
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

  // and its textures
  u_tex_ = u_->genTexture(false, cudaFilterModeLinear);
  u_prev_tex_ =  u_prev_->genTexture(false, cudaFilterModeLinear);
  u0_tex_ =  u0_->genTexture(false, cudaFilterModeLinear);
  pu_tex_ =  pu_->genTexture(false, cudaFilterModeLinear);
  q_tex_ =  q_->genTexture(false, cudaFilterModeLinear);
  ix_tex_ =  ix_->genTexture(false, cudaFilterModeLinear);
  it_tex_ =  it_->genTexture(false, cudaFilterModeLinear);

}

//------------------------------------------------------------------------------
void StereoCtFWarpingLevelHuber::init()
{
  u_->setValue(0.0f);
  u_prev_->setValue(0.0f);
  u0_->setValue(0.0f);
  pu_->setValue(0.0f);
  q_->setValue(0.0f);
}

//------------------------------------------------------------------------------
void StereoCtFWarpingLevelHuber::init(const StereoCtFWarpingLevel& rhs)
{
  const StereoCtFWarpingLevelHuber* from =
      dynamic_cast<const StereoCtFWarpingLevelHuber*>(&rhs);

  float inv_sf = params_->ctf.scale_factor; // >1 for adapting prolongated disparities

  if(params_->ctf.apply_median_filter)
  {
    imp::cu::filterMedian3x3(from->u0_.get(), from->u_.get());
    imp::cu::resample(u_.get(), from->u0_.get(), imp::InterpolationMode::linear, false);
  }
  imp::cu::resample(u_.get(), from->u_.get(), imp::InterpolationMode::linear, false);
  *u_ *= inv_sf;

  imp::cu::resample(pu_.get(), from->pu_.get(), imp::InterpolationMode::linear, false);
  imp::cu::resample(q_.get(), from->q_.get(), imp::InterpolationMode::linear, false);
}

//------------------------------------------------------------------------------
void StereoCtFWarpingLevelHuber::solve(std::vector<ImagePtr> images)
{
  std::cout << "StereoCtFWarpingLevelHuber: solving level " << level_ << " with " << images.size() << " images" << std::endl;

  // image textures

  i1_tex_ = images.at(0)->genTexture(false, cudaFilterModeLinear);
  i2_tex_ = images.at(1)->genTexture(false, cudaFilterModeLinear);

  // constants
//  constexpr float tau = 0.95f;
//  constexpr float sigma = 0.95f;
  float lin_step = 0.5f;

  // precond
//  constexpr float eta = 2.0f;

  // warping
  for (std::uint32_t warp = 0; warp < params_->ctf.warps; ++warp)
  {
    u_->copyTo(*u0_);

    // warping + gradients computation
    // TODO

    // compute preconditioner
    // TODO

    for (std::uint32_t iter = 0; iter < params_->ctf.iters; ++iter)
    {
      // solve dual kernel
      // TODO


      // and primal kernel
      // TODO

    } // iters
    lin_step /= 1.2f;

  } // warps
  IMP_CUDA_CHECK();
}



} // namespace cu
} // namespace imp


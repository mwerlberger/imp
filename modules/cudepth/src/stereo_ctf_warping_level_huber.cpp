#include <imp/cudepth/stereo_ctf_warping_level_huber.hpp>

#include <imp/cudepth/variational_stereo_parameters.hpp>
#include <imp/cuimgproc/cu_image_filter.cuh>
#include <imp/cuimgproc/cu_image_transform.cuh>

namespace imp {
namespace cu {

//------------------------------------------------------------------------------
StereoCtFWarpingLevelHuber::~StereoCtFWarpingLevelHuber()
{
  // thanks to smart pointers
}

//------------------------------------------------------------------------------
StereoCtFWarpingLevelHuber::StereoCtFWarpingLevelHuber(const std::shared_ptr<Parameters>& params, imp::Size2u size, size_type level)
  : StereoCtFWarpingLevel(params, size, level)
{
  u_.reset(new Image(size));
  u_prev_.reset(new Image(size));
  u0_.reset(new Image(size));
  pu_.reset(new Dual(size));
  q_.reset(new Image(size));
  ix_.reset(new Image(size));
  it_.reset(new Image(size));
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
  const StereoCtFWarpingLevelHuber* from = dynamic_cast<const StereoCtFWarpingLevelHuber*>(&rhs);

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

}



} // namespace cu
} // namespace imp


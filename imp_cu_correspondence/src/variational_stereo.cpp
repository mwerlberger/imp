#include <imp/cu_correspondence/variational_stereo.hpp>

#include <imp/cu_correspondence/stereo_ctf_warping.hpp>

namespace imp {
namespace cu {

//------------------------------------------------------------------------------
VariationalStereo::VariationalStereo(ParametersPtr params)
{
  if (params)
    params_ = params;
  else
    params_.reset(new Parameters());

  ctf_.reset(new StereoCtFWarping(params_));

}

//------------------------------------------------------------------------------
VariationalStereo::~VariationalStereo()
{

}


//------------------------------------------------------------------------------
void VariationalStereo::addImage(ConstImagePtrRef image)
{
  ctf_->addImage(image);
}

//------------------------------------------------------------------------------
void VariationalStereo::solve()
{
  ctf_->solve();
}

//------------------------------------------------------------------------------
VariationalStereo::ImagePtr VariationalStereo::getDisparities(size_type level)
{
  return ctf_->getDisparities(level);
}


} // namespace cu
} // namespace imp


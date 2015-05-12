#include <imp/cu_correspondence/variational_stereo.hpp>

#include <imp/cu_correspondence/stereo_ctf_warping.hpp>

namespace imp {
namespace cu {

//------------------------------------------------------------------------------
VariationalStereo::VariationalStereo(Parameters::Ptr params)
{
  if (params)
  {
    params_ = params;
  }
  else
  {
    params_ = std::make_shared<Parameters>();
  }

  ctf_.reset(new StereoCtFWarping(params_));
}


//------------------------------------------------------------------------------
VariationalStereo::~VariationalStereo()
{ ; }


//------------------------------------------------------------------------------
void VariationalStereo::addImage(const Image::Ptr& image)
{
  ctf_->addImage(image);
}


//------------------------------------------------------------------------------
void VariationalStereo::solve()
{
  ctf_->solve();
}


//------------------------------------------------------------------------------
VariationalStereo::Image::Ptr VariationalStereo::getDisparities(size_type level)
{
  return ctf_->getDisparities(level);
}


//------------------------------------------------------------------------------
VariationalStereo::Image::Ptr VariationalStereo::getOcclusion(size_type level)
{
  return ctf_->getOcclusion(level);
}


} // namespace cu
} // namespace imp


#include <imp/cudepth/variational_stereo.hpp>


#include <imp/cudepth/stereo_ctf_warping.hpp>



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
void VariationalStereo::addImage(ImagePtr image)
{
  ctf_->addImage(image);
}

//------------------------------------------------------------------------------
void VariationalStereo::solve()
{
  ctf_->solve();
}


} // namespace cu
} // namespace imp


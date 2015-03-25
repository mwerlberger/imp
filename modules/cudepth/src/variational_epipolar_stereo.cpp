#include <imp/cudepth/variational_epipolar_stereo.hpp>

#include <imp/cudepth/stereo_ctf_warping.hpp>

namespace imp {
namespace cu {

//------------------------------------------------------------------------------
VariationalEpipolarStereo::VariationalEpipolarStereo(ParametersPtr params)
  : VariationalStereo(params)
{
}

//------------------------------------------------------------------------------
VariationalEpipolarStereo::~VariationalEpipolarStereo()
{
}

//------------------------------------------------------------------------------
void VariationalEpipolarStereo::setCorrespondenceGuess(ConstVectorImagePtr disp)
{
  ctf_->setCorrespondenceGuess(disp);
}

//------------------------------------------------------------------------------
void VariationalEpipolarStereo::setEpiVecs(ConstVectorImagePtr epi_vec)
{
  ctf_->setEpiVecs(epi_vec);

}


} // namespace cu
} // namespace imp


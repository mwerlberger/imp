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
void VariationalEpipolarStereo::setFundamentalMatrix(const cu::Matrix3f& F)
{
  ctf_->setFundamentalMatrix(F);
}


//------------------------------------------------------------------------------
void VariationalEpipolarStereo::setIntrinsics(const cu::PinholeCamera& cam)
{
  ctf_->setIntrinsics(cam);
}

//------------------------------------------------------------------------------
void VariationalEpipolarStereo::setExtrinsics(const cu::SE3<float>& T_mov_fix)
{
  ctf_->setExtrinsics(T_mov_fix);
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


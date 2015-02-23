#include <imp/cudepth/stereo_ctf_warping_level.hpp>


namespace imp {
namespace cu {

////------------------------------------------------------------------------------
//StereoCtFWarpingLevel::StereoCtFWarpingLevel()
//{
//}

////------------------------------------------------------------------------------
//StereoCtFWarpingLevel::~StereoCtFWarpingLevel()
//{
//}


//------------------------------------------------------------------------------
StereoCtFWarpingLevel::StereoCtFWarpingLevel(std::shared_ptr<Parameters> params,
                                             imp::Size2u size, std::uint16_t level)
  : params_(params)
  , size_(size)
  , level_(level)
{
}

} // namespace cu
} // namespace imp

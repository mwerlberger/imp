#include <imp/cudepth/stereo_ctf_warping.hpp>

#include <memory>
#include <imp/cudepth/stereo_ctf_warping_level_huber.hpp>

//#include <imp/cudepth/variational_stereo_parameters.hpp>
//#include <imp/cucore/cu_image_gpu.cuh>
//#include <imp/cuimgproc/image_pyramid.hpp>

namespace imp {
namespace cu {

//------------------------------------------------------------------------------
StereoCtFWarping::StereoCtFWarping(std::shared_ptr<Parameters> params)
  : params_(params)
{
}

//------------------------------------------------------------------------------
StereoCtFWarping::~StereoCtFWarping()
{
  // thanks to managed ptrs
}

//------------------------------------------------------------------------------
void StereoCtFWarping::addImage(ImagePtr image)
{
  // generate image pyramid
  ImagePyramidPtr pyr(new ImagePyramid(image, params_->ctf.scale_factor));

  // update number of levels
  if (params_->ctf.levels > pyr->numLevels())
    params_->ctf.levels = pyr->numLevels();
  if (params_->ctf.coarsest_level > params_->ctf.levels - 1)
    params_->ctf.coarsest_level = params_->ctf.levels - 1;

  images_.push_back(image);
  image_pyramids_.push_back(pyr);

  std::cout << "we have now " << images_.size() << " images and "
            <<  image_pyramids_.size() << " pyramids in the CTF instance. "
             << "params_->ctf.levels: " << params_->ctf.levels
             << " (" << params_->ctf.coarsest_level << " -> " << params_->ctf.finest_level << ")"
             << std::endl;

  if (levels_.empty())
  {
    this->init();
  }
}

//------------------------------------------------------------------------------
void StereoCtFWarping::init()
{
  if (image_pyramids_.empty())
  {
    throw Exception("No Image set, can't initialize when number of levels is unknown.",
                    __FILE__, __FUNCTION__, __LINE__);
  }

  // just in case
  levels_.clear();

  for (size_type i=params_->ctf.finest_level; i<params_->ctf.coarsest_level; ++i)
  {
    Size2u sz = image_pyramids_.front()->size(i);
    levels_.emplace_back(new StereoCtFWarpingLevelHuber(params_, sz, i));
  }

}


} // namespace cu
} // namespace imp


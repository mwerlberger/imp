#include <imp/cudepth/stereo_ctf_warping.hpp>

#include <memory>

#include <imp/cudepth/stereo_ctf_warping_level_huber.cuh>
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
void StereoCtFWarping::init()
{
  if (image_pyramids_.empty())
  {
    throw Exception("No Image set, can't initialize when number of levels is unknown.",
                    __FILE__, __FUNCTION__, __LINE__);
  }

  // just in case
  levels_.clear();

  for (size_type i=params_->ctf.finest_level; i<=params_->ctf.coarsest_level; ++i)
  {
    Size2u sz = image_pyramids_.front()->size(i);
    levels_.emplace_back(new StereoCtFWarpingLevelHuber(params_, sz, i));
  }
}

//------------------------------------------------------------------------------
bool StereoCtFWarping::ready()
{
  // check if all vectors are of the same length and not empty
  size_type desired_num_levels =
      params_->ctf.coarsest_level - params_->ctf.finest_level + 1;

  if (images_.empty() || image_pyramids_.empty() || levels_.empty() ||
      params_->ctf.coarsest_level <= params_->ctf.finest_level ||
      images_.size() < 2 || // at least two images -> maybe adapt to the algorithm?
      image_pyramids_.front()->numLevels() != desired_num_levels ||
      levels_.size() != desired_num_levels
      )
  {
    return false;
  }
  return true;
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
void StereoCtFWarping::solve()
{
  if (!this->ready())
  {
    throw Exception("not initialized correctly; bailing out.",
                    __FILE__, __FUNCTION__, __LINE__);
  }

  // the image vector that is used as input for the level solvers
  std::vector<ImagePtr> lev_images;


  // the first level is initialized differently so we solve this one first
  size_type lev = params_->ctf.coarsest_level;
  levels_.at(lev)->init();
  // gather images of current scale level
  for (auto pyr : image_pyramids_)
  {
    lev_images.push_back(std::dynamic_pointer_cast<Image>(pyr->at(lev)));
  }
  levels_.at(lev)->solve(lev_images);
  lev_images.clear();
  // and then loop until we reach the finest level
  // note that we loop with +1 idx as we would result in a buffer underflow
  // due to operator-- on size_type which is an unsigned type.
  for (; lev > params_->ctf.finest_level; --lev)
  {
    levels_.at(lev-1)->init(*levels_.at(lev));

    // gather images of current scale level
    for (auto pyr : image_pyramids_)
    {
      lev_images.push_back(std::dynamic_pointer_cast<Image>(pyr->at(lev-1)));
    }
    levels_.at(lev-1)->solve(lev_images);
    lev_images.clear();
  }

}

} // namespace cu
} // namespace imp


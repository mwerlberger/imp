#include <imp/cudepth/stereo_ctf_warping.hpp>

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
void StereoCtFWarping::addImage(ImagePtr image)
{
  // generate image pyramid
  ImagePyramidPtr pyr(new ImagePyramid(image, params_->ctf.scale_factor));
  images_.push_back(image);
  image_pyramids_.push_back(pyr);

  std::cout << "we have now " << images_.size() << " images and "
            <<  image_pyramids_.size() << " pyramids in the CTF instance."
             << std::endl;
}



} // namespace cu
} // namespace imp


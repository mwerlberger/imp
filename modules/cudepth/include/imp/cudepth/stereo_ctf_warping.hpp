#ifndef IMP_CU_STEREOCTFWARPING_HPP
#define IMP_CU_STEREOCTFWARPING_HPP

#include <memory>
#include <vector>

#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cuimgproc/image_pyramid.hpp>

#include <imp/cudepth/variational_stereo_parameters.hpp>
//#include <imp/cudepth/stereo_ctf_warping_level.hpp>


namespace imp {
namespace cu {

// forward declarations
class StereoCtFWarpingLevel;

/**
 * @brief The StereoCtFWarping class
 */
class StereoCtFWarping
{
public:
  using Parameters = VariationalStereoParameters;

  using Image = imp::cu::ImageGpu32fC1;
  using ImagePtr = std::shared_ptr<Image>;
  using ConstImagePtrRef = const std::shared_ptr<Image>&;

  using ImagePyramid = imp::ImagePyramid32fC1;
  using ImagePyramidPtr = std::shared_ptr<ImagePyramid>;

public:
  StereoCtFWarping() = delete;
  virtual ~StereoCtFWarping();// = default;
//  StereoCtFWarping(const StereoCtFWarping&);
//  StereoCtFWarping(StereoCtFWarping&&);

  StereoCtFWarping(std::shared_ptr<Parameters> params);
  void init();

  void addImage(ImagePtr image);
  void solve();

  // don't we wanna have this in a vector type?
  ImagePtr getU(std::uint32_t level);
  ImagePtr getV(std::uint32_t level);

private:
  std::shared_ptr<Parameters> params_; //!< configuration parameters
  std::vector<ImagePtr> images_; //!< all unprocessed input images
  std::vector<ImagePyramidPtr> image_pyramids_; //!< image pyramids corresponding to the unprocesed input images
  std::vector<std::unique_ptr<StereoCtFWarpingLevel>> levels_;
};

} // namespace cu
} // namespace imp

#endif // IMP_CU_STEREOCTFWARPING_HPP

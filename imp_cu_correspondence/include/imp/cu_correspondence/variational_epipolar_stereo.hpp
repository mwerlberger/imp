#ifndef IMP_CU_VARIATIONAL_EPIPOLAR_STEREO_HPP
#define IMP_CU_VARIATIONAL_EPIPOLAR_STEREO_HPP


#include <cstdint>
#include <memory>

#include <imp/cu_core/cu_matrix.cuh>
#include <imp/cu_core/cu_se3.cuh>
#include <imp/cu_core/cu_pinhole_camera.cuh>
#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/cu_correspondence/variational_stereo_parameters.hpp>
#include <imp/cu_correspondence/variational_stereo.hpp>

namespace imp {
namespace cu {

/**
 * @brief The Stereo class takes an image pair with known epipolar geometry
 *        (fundamental matrix) and estimates the disparity map
 */
//template<typename Pixel, imp::PixelType pixel_type>
class VariationalEpipolarStereo : public VariationalStereo
{
public:
  //! @todo (MWE) first do the implementation with specific type (32fC1) and later generalize
//  using Image = imp::cu::ImageGpu<Pixel, pixel_type>;
  using Image = imp::cu::ImageGpu32fC1;
  using ImagePtr = std::shared_ptr<Image>;
  using ConstImagePtrRef = const std::shared_ptr<Image>&;

  using VectorImage = imp::cu::ImageGpu32fC2;
  using VectorImagePtr = std::shared_ptr<VectorImage>;
  using ConstVectorImagePtr = const std::shared_ptr<VectorImage>&;

  using Cameras = std::vector<cu::PinholeCamera>;


  using Parameters = VariationalStereoParameters;
  using ParametersPtr = std::shared_ptr<Parameters>;

public:
  VariationalEpipolarStereo(ParametersPtr params=nullptr);
  virtual ~VariationalEpipolarStereo(); //= default;

//  virtual void setTransformation
  virtual void setFundamentalMatrix(const cu::Matrix3f& F);
  virtual void setIntrinsics(const Cameras& cams);
  virtual void setExtrinsics(const cu::SE3<float>& T_mov_fix);
  virtual void setDepthProposal(ConstImagePtrRef depth_proposal,
                                ConstImagePtrRef depth_proposal_sigma2=nullptr);



private:
};




} // namespace cu
} // namespace imp

#endif // IMP_CU_STEREO_HPP

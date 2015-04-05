#ifndef IMP_CU_VARIATIONAL_EPIPOLAR_STEREO_HPP
#define IMP_CU_VARIATIONAL_EPIPOLAR_STEREO_HPP


#include <cstdint>
#include <memory>

#include <imp/cucore/cu_matrix.cuh>
#include <imp/cucore/cu_se3.cuh>
#include <imp/cucore/cu_pinhole_camera.cuh>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cudepth/variational_stereo_parameters.hpp>
#include <imp/cudepth/variational_stereo.hpp>

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

  using Parameters = VariationalStereoParameters;
  using ParametersPtr = std::shared_ptr<Parameters>;

public:
  VariationalEpipolarStereo(ParametersPtr params=nullptr);
  virtual ~VariationalEpipolarStereo(); //= default;

//  virtual void setTransformation
  virtual void setFundamentalMatrix(const cu::Matrix3f& F);
  virtual void setIntrinsics(const cu::PinholeCamera& cam);
  virtual void setExtrinstics(const cu::SE3<float>& T_mov_fix);
  virtual void setCorrespondenceGuess(ConstVectorImagePtr disp);
  virtual void setEpiVecs(ConstVectorImagePtr epi_vec);



private:
};




} // namespace cu
} // namespace imp

#endif // IMP_CU_STEREO_HPP

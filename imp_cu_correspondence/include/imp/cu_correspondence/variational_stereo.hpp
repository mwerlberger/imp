#ifndef IMP_CU_VARIATIONAL_STEREO_HPP
#define IMP_CU_VARIATIONAL_STEREO_HPP


#include <cstdint>
#include <memory>

#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/cu_correspondence/variational_stereo_parameters.hpp>

namespace imp {
namespace cu {

//// forward declarations
//template<typename Pixel, imp::PixelType pixel_type>
//class ImageGpu<Pixel, pixel_type>;

class StereoCtFWarping;


/**
 * @brief The Stereo class takes a stereo image pair and estimates the disparity map
 */
//template<typename Pixel, imp::PixelType pixel_type>
class VariationalStereo
{
public:
  //! @todo (MWE) first do the implementation with specific type (32fC1) and later generalize
//  using Image = imp::cu::ImageGpu<Pixel, pixel_type>;
  using Image = imp::cu::ImageGpu32fC1;
  using ImagePtr = std::shared_ptr<Image>;
  using ConstImagePtrRef = const std::shared_ptr<Image>&;
  using Parameters = VariationalStereoParameters;
  using ParametersPtr = std::shared_ptr<Parameters>;

public:
  VariationalStereo(ParametersPtr params=nullptr);
  virtual ~VariationalStereo(); //= default;

  virtual void addImage(ConstImagePtrRef image);
  virtual void solve();

  virtual ImagePtr getDisparities(size_type level=0);
  virtual ImagePtr getOcclusion(size_type level=0);

  // getters / setters
  virtual inline ParametersPtr parameters() {return params_;}

protected:
  ParametersPtr params_;  //!< configuration parameters
  std::unique_ptr<StereoCtFWarping> ctf_;  //!< performing a coarse-to-fine warping scheme
};

} // namespace cu
} // namespace imp

#endif // IMP_CU_VARIATIONAL_STEREO_HPP

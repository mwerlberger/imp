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
class VariationalStereo
{
public:
  //! @todo (MWE) first do the implementation with specific type (32fC1) and later generalize
  using Image = imp::cu::ImageGpu32fC1;
  using Parameters = VariationalStereoParameters;

public:
  VariationalStereo(Parameters::Ptr params=nullptr);
  virtual ~VariationalStereo(); //= default;

  virtual void addImage(const Image::Ptr& image);
  virtual void solve();

  virtual Image::Ptr getDisparities(size_type level=0);
  virtual Image::Ptr getOcclusion(size_type level=0);

  // getters / setters
  virtual inline Parameters::Ptr parameters() {return params_;}

protected:
  Parameters::Ptr params_;  //!< configuration parameters
  std::unique_ptr<StereoCtFWarping> ctf_;  //!< performing a coarse-to-fine warping scheme
};

} // namespace cu
} // namespace imp

#endif // IMP_CU_VARIATIONAL_STEREO_HPP

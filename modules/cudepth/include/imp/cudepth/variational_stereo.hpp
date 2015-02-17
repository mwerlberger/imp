#ifndef IMP_CU_STEREO_HPP
#define IMP_CU_STEREO_HPP

#include <imp/cudepth/stereo_solver_enum.hpp>

#include <cstdint>

namespace imp {
namespace cu {

// forward declarations
template<typename Pixel, imp::PixelType pixel_type>
class ImageGpu<Pixel, pixel_type>;

class CTFWarpingSolver;

/**
 * @brief The Stereo class takes a stereo image pair and estimates the disparity map
 */
template<typename Pixel, imp::PixelType pixel_type>
class VariationalStereo
{
public:
  using Image = imp::cu::ImageGpu<Pixel, pixel_type>;
  using ImagePtr = std::shared_ptr<Image>;
  using ConstImagePtrRef = const std::shared_ptr<Image>&;


  // nested parameter struct combining all available settings
  struct Parameters
  {
    int verbose=0; //!< verbosity level (the higher, the more the Stereo algorithm talks to us)
    StereoPDSolver solver=StereoPDSolver::HuberL1; //!< selected primal-dual solver / model
    float lambda = 50.0f; //!< tradeoff between regularization and matching term

    struct CTF // we might want to define this externally for all ctf approaches?
    {
      float scale_factor = 0.5f; //!< multiplicative scale factor between coarse-to-fine pyramid levels
      std::uint32_t iters = 100;
      std::uint32_t warps =  10;
      std::uint32_t levels= UINT32_MAX;
      std::uint32_t coarsest_level = UINT32_MAX;
      std::uint32_t finest_level = UINT32_MAX;
      bool apply_median_filter = true;
    };

    CTF ctf;

    friend std::ostream& operator<<(std::ostream& stream, const Parameters& p);

  };

public:
  VariationalStereo();
  virtual ~VariationalStereo() = default;

  void addImage(ConstImagePtrRef image);

  std::unique_ptr<CTFWarpingSolver> ctf_;

private:

  Parameters params_;
};

} // namespace cu
} // namespace imp

#endif // IMP_CU_STEREO_HPP

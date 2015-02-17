#ifndef IMP_CU_STEREO_CTF_WARPING_LEVEL_HPP
#define IMP_CU_STEREO_CTF_WARPING_LEVEL_HPP

#include <cstdint>

#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/core/size.hpp>


namespace imp {
namespace cu {

// forward decl
class VariationalStereoParameters;


/**
 * @brief The StereoCtFWarpingLevel class
 */
class StereoCtFWarpingLevel
{
public:
  using Parameters = VariationalStereoParameters;
  using Image = imp::cu::ImageGpu32fC1;
  using ImagePtr = std::shared_ptr<Image>;



public:
  StereoCtFWarpingLevel() = delete;
  virtual ~StereoCtFWarpingLevel() = default;

  StereoCtFWarpingLevel(std::shared_ptr<Parameters> params,
                        imp::Size2u size, std::uint16_t level)
    : params_(params)
    , size_(size)
    , level_(level)
  {
  }

  virtual void init() = 0;
  virtual void init(const StereoCtFWarpingLevel& from) = 0;
  virtual void solve(std::vector<ImagePtr> images) = 0;


  // setters / getters
  inline imp::Size2u size() { return size_; }
  inline std::uint16_t level() { return level_; }

protected:
  std::shared_ptr<Parameters> params_; //!< configuration parameters
  imp::Size2u size_;
  std::uint16_t level_; //!< level number in the ctf pyramid (0=finest .. n=coarsest)
};

} // namespace cu
} // namespace imp

#endif // IMP_CU_STEREO_CTF_WARPING_LEVEL_HPP

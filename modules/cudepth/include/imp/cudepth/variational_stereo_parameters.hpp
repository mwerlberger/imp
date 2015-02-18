#ifndef VARIATIONAL_STEREO_PARAMETERS_HPP
#define VARIATIONAL_STEREO_PARAMETERS_HPP

#include <cstdint>
#include <imp/core/types.hpp>
#include <imp/cudepth/stereo_solver_enum.hpp>

namespace imp {
namespace cu {

// the parameter struct
struct VariationalStereoParameters
{
  int verbose=0; //!< verbosity level (the higher, the more the Stereo algorithm talks to us)
  StereoPDSolver solver=StereoPDSolver::HuberL1; //!< selected primal-dual solver / model
  float lambda = 50.0f; //!< tradeoff between regularization and matching term

  // settings for the ctf warping
  struct CTF // we might want to define this externally for all ctf approaches?
  {
    float scale_factor = 0.5f; //!< multiplicative scale factor between coarse-to-fine pyramid levels
    std::uint32_t iters = 100;
    std::uint32_t warps =  10;
    size_type levels = UINT32_MAX;
    size_type coarsest_level = UINT32_MAX;
    size_type finest_level = 0;
    bool apply_median_filter = true;
  };

  CTF ctf;

  friend std::ostream& operator<<(std::ostream& stream,
                                  const VariationalStereoParameters& p);
};


} // namespace cu
} // namespace imp

#endif // VARIATIONAL_STEREO_PARAMETERS_HPP


#ifndef IMP_CU_K_STEREO_CTF_WARPING_LEVEL_HUBER_CUH
#define IMP_CU_K_STEREO_CTF_WARPING_LEVEL_HUBER_CUH

#include <cuda_runtime_api.h>
#include <imp/core/types.hpp>
#include <imp/cucore/cu_utils.hpp>
#include <imp/cucore/cu_k_derivative.cuh>
#include <imp/cuda_toolkit/helper_math.h>


namespace imp {
namespace cu {

//-----------------------------------------------------------------------------
/** restricts the udpate to +/- lin_step around the given value in lin_tex
 * @note \a d_srcdst and the return value is identical.
 * @todo (MWE) move function to common kernel def file for all stereo models
 */
template<typename Pixel>
__device__ Pixel k_linearized_update(Pixel& d_srcdst, Texture2D& lin_tex,
                                     const float lin_step,
                                     const int x, const int y)
{
  Pixel lin = lin_tex.fetch<Pixel>(x, y);
  d_srcdst = max(lin-lin_step,
                 min(lin+lin_step, d_srcdst));
  return d_srcdst;
}

//-----------------------------------------------------------------------------
/**
 * @brief k_primalUpdate is the Huber-L1-Precondition model's primal update kernel
 * @note PPixel and DPixel denote for the Pixel type/dimension of primal and dual variable
 */
template<typename PPixel>
__global__ void k_primalUpdate(PPixel* d_u, PPixel* d_u_prev, const size_type stride,
                               std::uint32_t width, std::uint32_t height,
                               const float lambda, const float tau,
                               const float lin_step,
                               Texture2D u_tex, Texture2D u0_tex,
                               Texture2D pu_tex, Texture2D ix_tex, Texture2D it_tex)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x<width && y<height)
  {
    float u0 = u0_tex.fetch<float>(x,y);
    float it = it_tex.fetch<float>(x, y);
    float ix = ix_tex.fetch<float>(x, y);

    // divergence operator (dpAD) of dual var
    float div = dpAd(pu_tex, x, y, width, height);

    // save current u
    float u_prev = u_tex.fetch<float>(x, y);
    float u = u_prev;
    u += tau*div;

    // prox operator
    float prox = it + (u-u0)*ix;
    prox /= max(1e-9f, ix*ix);
    float tau_lambda = tau*lambda;

    if(prox < -tau_lambda)
    {
      u += tau_lambda*ix;
    }
    else if(prox > tau_lambda)
    {
      u -= tau_lambda*ix;
    }
    else if (std::abs(prox) <= tau_lambda)
    {
      u -= prox*ix;
    }

    // restrict update step because of linearization only valid in small neighborhood
    u = k_linearized_update(u, u0_tex, lin_step, x, y);

    d_u[y*stride+x] = u;
    d_u_prev[y*stride+x] = 2.f*u - u_prev;
  }
}

//-----------------------------------------------------------------------------
template<typename DPixel>
__global__ void k_dualUpdate(DPixel* d_pu, const size_type stride_pu,
                             std::uint32_t width, std::uint32_t height,
                             const float eps_u, const float sigma,
                             Texture2D u_prev_tex, Texture2D pu_tex)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x<width && y<height)
  {
    // update pu
    float2 du = dp(u_prev_tex, x, y);
    float2 pu = pu_tex.fetch<float2>(x,y);
    pu  = (pu + sigma*du) / (1.f + sigma*eps_u);
    pu = pu / max(1.0f, length(pu));

    d_pu[y*stride_pu+x] = {pu.x, pu.y};
  }
}


} // namespace cu
} // namespace imp



#endif // IMP_CU_K_STEREO_CTF_WARPING_LEVEL_HUBER_CUH


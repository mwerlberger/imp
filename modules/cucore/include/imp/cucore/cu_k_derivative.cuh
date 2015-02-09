#ifndef IMP_CU_K_DERIVATIVE_CUH
#define IMP_CU_K_DERIVATIVE_CUH

#include <cuda_runtime.h>

namespace imp {
namespace cu {

//-----------------------------------------------------------------------------
/** compute forward differences in x- and y- direction */
static __device__ __forceinline__ float2 dp(
    const imp::cu::Texture2D& tex, size_t x, size_t y)
{
  float2 grad = make_float2(0.0f, 0.0f);
  float cval = tex2D<float>(tex, x+.5f, y+.5f);
  grad.x = tex2D<float>(tex, x+1.5f, y+0.5f) - cval;
  grad.y = tex2D<float>(tex, x+0.5f, y+1.5f) - cval;
  return grad;
}

//-----------------------------------------------------------------------------
/** compute divergence using backward differences (adjugate from dp). */
static __device__ __forceinline__
float dpAd(const imp::cu::Texture2D& tex,
           size_t x, size_t y, size_t width, size_t height)
{
  float2 cval = tex2D<float2>(tex, x+0.5f, y+0.5f);
  float2 wval = tex2D<float2>(tex, x-0.5f, y+0.5f);
  float2 nval = tex2D<float2>(tex, x+0.5f, y-0.5f);

  if (x == 0)
    wval.x = 0.0f;
  else if (x >= width-1)
    cval.x = 0.0f;

  if (y == 0)
    nval.y = 0.0f;
  else if (y >= height-1)
    cval.y = 0.0f;

  return (cval.x - wval.x + cval.y - nval.y);
}

} // namespace cu
} // namespace imp

#endif // IMP_CU_K_DERIVATIVE_CUH


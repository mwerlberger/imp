#ifndef IMP_CU_K_WARPED_GRADIENTS_CUH
#define IMP_CU_K_WARPED_GRADIENTS_CUH

#include <cuda_runtime_api.h>
#include <imp/core/types.hpp>



namespace imp {
namespace cu {

template<typename Pixel>
__global__ void k_warpedGradients(Pixel* ix, Pixel* it, size_type stride,
                                  std::uint32_t width, std::uint32_t height,
//                                  std::uint32_t roi_x, std::uint32_t roi_y,
                                  Texture2D i1_tex, Texture2D i2_tex, Texture2D u0_tex)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x /*+ roi_x*/;
  const int y = blockIdx.y*blockDim.y + threadIdx.y /*+ roi_y*/;
  const int c = y*stride+x;

  if (x<width && y<height)
  {
    float disparity = u0_tex.fetch<float>(x,y);
    float wx = x+disparity;

    float bd = 0.5f;
//    if ((wx < bd) || (x < bd) || (wx > width-bd) || (x > width-bd) ||
//        (y < bd) || (y > height-bd))
    /// @todo (MWE) check border handling!
    if (wx<1 || wx>width-2)
    {
      ix[c] =  0.0f;
      it[c] =  0.0f;
    }
    else
    {
      Pixel i1_c;
      Pixel i2_w_c, i2_w_m, i2_w_p;

      i1_tex.fetch(i1_c, x, y);

      i2_tex.fetch(i2_w_c, wx, y);
      i2_tex.fetch(i2_w_m, wx-0.5f, y);
      i2_tex.fetch(i2_w_p, wx+0.5f, y);

      // spatial gradient on warped image
      ix[c] = i2_w_p - i2_w_m;
      // temporal gradient
      it[c] = i2_w_c - i1_c;
    }

  }

}


} // namespace cu
} // namespace imp



#endif // IMP_CU_K_WARPED_GRADIENTS_CUH


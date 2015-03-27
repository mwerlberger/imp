#ifndef IMP_CU_K_WARPED_GRADIENTS_CUH
#define IMP_CU_K_WARPED_GRADIENTS_CUH

#include <cuda_runtime_api.h>
#include <imp/core/types.hpp>
#include <imp/cuda_toolkit/helper_math.h>


namespace imp {
namespace cu {

//------------------------------------------------------------------------------
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

    float bd = .5f;
    if ((wx < bd) || (x < bd) || (wx > width-bd-1) || (x > width-bd-1) ||
        (y<bd) || (y>height-bd-1))
    {
      ix[c] =  0.0f;
      it[c] =  0.0f;
    }
    else
    {
      Pixel i1_c, i2_w_c, i2_w_m, i2_w_p;

      i1_tex.fetch(i1_c, x, y);

      i2_tex.fetch(i2_w_c, wx, y);
      i2_tex.fetch(i2_w_m, wx-0.5f, y);
      i2_tex.fetch(i2_w_p, wx+0.5f, y);

      // spatial gradient on warped image
      ix[c] = i2_w_p - i2_w_m;
      // temporal gradient between the warped moving image and the fixed image
      it[c] = i2_w_c - i1_c;
    }

  }
}

//------------------------------------------------------------------------------
template<typename Pixel>
__global__ void k_warpedGradientsEpipolarConstraint(
    Pixel* iw, Pixel* ix, Pixel* it, size_type stride,
    std::uint32_t width, std::uint32_t height,
    // std::uint32_t roi_x, std::uint32_t roi_y,
    Texture2D i1_tex, Texture2D i2_tex, Texture2D u0_tex,
    Texture2D correspondence_guess_tex, Texture2D epi_vec_tex)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x /*+ roi_x*/;
  const int y = blockIdx.y*blockDim.y + threadIdx.y /*+ roi_y*/;
  const int c = y*stride+x;

  if (x<width && y<height)
  {
    float2 pt_p = correspondence_guess_tex.fetch<float2>(x,y);
    float2 epi_vec = epi_vec_tex.fetch<float2>(x,y);

    float disparity = u0_tex.fetch<float>(x,y);
    float2 w_pt_p = pt_p + normalize(epi_vec)*disparity; // assuming that epi_vec is the unit vec

    float bd = .5f;
    if ((w_pt_p.x < bd) || (x < bd) || (w_pt_p.y > width-bd-1) || (x > width-bd-1) ||
        (w_pt_p.y<bd) || (w_pt_p.y>height-bd-1) || (y<bd) || (y>height-bd-1))
    {
      ix[c] =  0.0f;
      it[c] =  0.0f;
    }
    else
    {
      Pixel i1_c, i2_w_c, i2_w_m, i2_w_p;

      i1_tex.fetch(i1_c, x, y);

       /// @todo (MWE) don't we want to have the gradient along the epipolar line??
      i2_tex.fetch(i2_w_c, w_pt_p.x, w_pt_p.y);
      i2_tex.fetch(i2_w_m, w_pt_p.x-0.5f*epi_vec.x, w_pt_p.y-0.5f*epi_vec.y);
      i2_tex.fetch(i2_w_p, w_pt_p.x+0.5f*epi_vec.x, w_pt_p.y+0.5f*epi_vec.y);

      // spatial gradient on warped image
      ix[c] = i2_w_p - i2_w_m;
      // temporal gradient between the warped moving image and the fixed image
      it[c] = i2_w_c - i1_c;
      // warped image
      iw[c] = i2_w_c;
    }

  }
}


} // namespace cu
} // namespace imp



#endif // IMP_CU_K_WARPED_GRADIENTS_CUH


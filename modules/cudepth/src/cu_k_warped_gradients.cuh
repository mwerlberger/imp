#ifndef IMP_CU_K_WARPED_GRADIENTS_CUH
#define IMP_CU_K_WARPED_GRADIENTS_CUH

#include <cuda_runtime_api.h>
#include <imp/core/types.hpp>
#include <imp/cuda_toolkit/helper_math.h>
#include <imp/cucore/cu_pinhole_camera.cuh>
#include <imp/cucore/cu_matrix.cuh>
#include <imp/cucore/cu_se3.cuh>


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
    cu::PinholeCamera cam1, cu::PinholeCamera cam2,
    const cu::Matrix3f F_ref_cur, const cu::SE3<float> T_mov_fix,
    Texture2D i1_tex, Texture2D i2_tex, Texture2D u0_tex,
    Texture2D depth_proposal_tex, Texture2D depth_proposal_sigma2_tex)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x /*+ roi_x*/;
  const int y = blockIdx.y*blockDim.y + threadIdx.y /*+ roi_y*/;
  const int c = y*stride+x;

  if (x<width && y<height)
  {
    // compute epipolar geometry
    float mu = depth_proposal_tex.fetch<float>(x,y);
//    float sigma = sqrtf(depth_proposal_sigma2_tex.fetch<float>(x,y));
    Vec32fC2 px_ref((float)x, (float)y);
    Vec32fC3 f_ref = normalize(cam1.cam2world(px_ref));
    Vec32fC2 px_mean = cam2.world2cam(T_mov_fix * (f_ref*mu));

    // check if current mean projects in image /*and mark if not*/
    if((px_mean.x >= width) || (px_mean.y >= height) || (px_mean.x < 0) || (px_mean.y < 0))
    {
      //d_converged[y*stride_32s+x] = -2;
      return;
    }

//    Vec32fC2 px_p3s = cam2.world2cam(T_mov_fix * (f_ref*(mu + 3.f*sigma)));
    Vec32fC3 px_mean_h(px_mean.x, px_mean.y, 1.0);
    Vec32fC3 epi_line = F_ref_cur*px_mean_h;
    Vec32fC2 epi_line_vec(epi_line.y, -epi_line.x);
    Vec32fC2 epi_vec = normalize(epi_line_vec);


//    float2 px_mean = correspondence_guess_tex.fetch<float2>(x,y);

    float disparity = u0_tex.fetch<float>(x,y);
    Vec32fC2 w_pt_p = px_mean + epi_vec*disparity; // assuming that epi_vec is the unit vec

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


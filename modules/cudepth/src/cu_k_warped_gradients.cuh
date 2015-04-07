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
    float2 px_ref = make_float2((float)x, (float)y);
    float3 px_ref_w = cam1.cam2world(px_ref);
    float3 f_ref = normalize(px_ref_w);
    float2 px_mean = cam2.world2cam(T_mov_fix * (f_ref*mu));

//    Vec32fC2 px_p3s = cam2.world2cam(T_mov_fix * (f_ref*(mu + 3.f*sigma)));
    float3 px_mean_h = make_float3(px_mean.x, px_mean.y, 1.0f);
    float3 epi_line = F_ref_cur*px_mean_h;
    float2 epi_line_vec = make_float2(epi_line.y, -epi_line.x);
    float2 epi_vec = normalize(epi_line_vec);

    if(x==128 && y==128)
    {
      printf("cam: %f, %f; %f, %f\n", cam1.fx(), cam1.fy(), cam1.cx(), cam1.cy());
      printf("px_ref_w: %f, %f, %f (length: %f)\n", px_ref_w.x, px_ref_w.y, px_ref_w.z, imp::length(px_ref_w));
      float3 n_foo = normalize(px_ref_w);
      printf("n_foo: %f, %f, %f (length: %f)\n", n_foo.x, n_foo.y, n_foo.z, imp::length(n_foo));

      printf("f_ref: %f, %f %f\n", f_ref.x, f_ref.y, f_ref.z);
      printf("px_mean: %f, %f (length: %f)\n", px_mean.x, px_mean.y, imp::length(px_mean));
      printf("epi_line: %f, %f %f\n", epi_line.x, epi_line.y, epi_line.z);

      printf("epi_line_vec: %f, %f (length: %f)\n", epi_line_vec.x, epi_line_vec.y, imp::length(epi_line_vec));
      printf("epi_vec: %f, %f (length: %f)\n\n", epi_vec.x, epi_vec.y, imp::length(epi_vec));
    }

//    float2 px_mean = correspondence_guess_tex.fetch<float2>(x,y);

    float disparity = u0_tex.fetch<float>(x,y);
    float2 w_pt_p = px_mean + epi_vec*disparity; // assuming that epi_vec is the unit vec

    float bd = .5f;
    // check if current mean projects in image /*and mark if not*/
    // and if warped point is within a certain image area
    if ((px_mean.x >= width) || (px_mean.y >= height) || (px_mean.x < 0) || (px_mean.y < 0) ||
        (w_pt_p.x < bd) || (x < bd) || (w_pt_p.y > width-bd-1) || (x > width-bd-1) ||
        (w_pt_p.y<bd) || (w_pt_p.y>height-bd-1) || (y<bd) || (y>height-bd-1))
    {
      ix[c] = 0.0f;
      it[c] = 0.0f;
      iw[c] = 0.0f;
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


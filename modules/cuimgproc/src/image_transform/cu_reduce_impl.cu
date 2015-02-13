#include <imp/cuimgproc/cu_image_transform.cuh>

#include <memory>
#include <cstdint>
#include <cmath>

#include <cuda_runtime.h>

#include <imp/core/types.hpp>
#include <imp/core/roi.hpp>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cucore/cu_utils.hpp>
#include <imp/cucore/cu_texture.cuh>

#ifndef IMP_CU_REDUCE_IMPL_CU
#define IMP_CU_REDUCE_IMPL_CU


namespace imp {
namespace cu {

//-----------------------------------------------------------------------------
template<typename Pixel>
__global__ void k_reduce(Pixel* d_dst, size_type stride,
                         std::uint32_t dst_width, std::uint32_t dst_height,
                         std::uint32_t roi_x, std::uint32_t roi_y,
                         float sf_x, float sf_y, Texture2D src_tex)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x + roi_x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y + roi_y;
  if (x<dst_width && y<dst_height)
  {
    Pixel val;
    src_tex.fetch(val, x, y, sf_x, sf_y);
    d_dst[y*stride+x] = val;
  }
}

//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
void reduce(ImageGpu<Pixel, pixel_type>* dst, ImageGpu<Pixel, pixel_type>* src,
            imp::InterpolationMode interp, bool gauss_prefilter)
{
  imp::Roi2u src_roi = src->roi();
  imp::Roi2u dst_roi = dst->roi();

  // scale factor for x/y > 0 && < 1 (for multiplication with dst coords in the kernel!)
  float sf_x = static_cast<float>(src_roi.width()) / static_cast<float>(dst_roi.width());
  float sf_y = static_cast<float>(src_roi.height()) / static_cast<float>(dst_roi.height());
  float sf = .5f*(sf_x+sf_y);

  if (gauss_prefilter)
  {
        float sigma = 1/(3*sf) ;  // empirical magic
        std::uint16_t kernel_size = std::ceil(6.0f*sigma);
        if (kernel_size % 2 == 0)
          kernel_size++;

    //! @todo (MWE) implement gaussian filter
        std::cerr << "(gpu) reduce: Gaussian prefilter not implemented! (continuing)" << std::endl;
  }

  cudaTextureFilterMode tex_filter_mode = (interp == InterpolationMode::linear) ?
        cudaFilterModeLinear : cudaFilterModePoint;
  if (src->bitDepth() < 32)
    tex_filter_mode = cudaFilterModePoint;
  std::unique_ptr<Texture2D> src_tex = src->genTexture(false, tex_filter_mode);

  std::cout << "sf: " << sf << "; roi_size: " << dst_roi.size() << std::endl;
  Fragmentation<16,16> dst_frag(dst_roi.size());

  switch(interp)
  {
  case InterpolationMode::point:
  case InterpolationMode::linear:
    // fallthrough intended
    k_reduce <<< dst_frag.dimGrid, dst_frag.dimBlock/*, 0, stream*/
        >>> (dst->data(), dst->stride(), dst->width(), dst->height(),
             dst_roi.x(), dst_roi.y(), sf_x , sf_y, *src_tex);
    break;
//  case InterpolationMode::cubic:
//    cuTransformCubicKernel_32f_C1
//        <<< dimGridOut, dimBlock, 0, stream >>> (dst->data(), dst->stride(), dst->width(), dst->height(),
//                                      sf_x , sf_y);
//    break;
//  case InterpolationMode::cubicSpline:
//    cuTransformCubicSplineKernel_32f_C1
//        <<< dimGridOut, dimBlock, 0, stream >>> (dst->data(), dst->stride(), dst->width(), dst->height(),
//                                      sf_x , sf_y);
//    break;
  }

  IMP_CUDA_CHECK();
}

//
// template instantiations for all our image types
//

//template void reduce(ImageGpu8uC1* dst, ImageGpu8uC1* src, imp::InterpolationMode interp, bool gauss_prefilter, cudaStream_t stream);
template void reduce(ImageGpu8uC1* dst, ImageGpu8uC1* src, InterpolationMode interp, bool gauss_prefilter);
template void reduce(ImageGpu8uC2* src, ImageGpu8uC2* dst, InterpolationMode interp, bool gauss_prefilter);
template void reduce(ImageGpu8uC4* src, ImageGpu8uC4* dst, InterpolationMode interp, bool gauss_prefilter);

template void reduce(ImageGpu16uC1* src, ImageGpu16uC1* dst, InterpolationMode interp, bool gauss_prefilter);
template void reduce(ImageGpu16uC2* src, ImageGpu16uC2* dst, InterpolationMode interp, bool gauss_prefilter);
template void reduce(ImageGpu16uC4* src, ImageGpu16uC4* dst, InterpolationMode interp, bool gauss_prefilter);

template void reduce(ImageGpu32sC1* src, ImageGpu32sC1* dst, InterpolationMode interp, bool gauss_prefilter);
template void reduce(ImageGpu32sC2* src, ImageGpu32sC2* dst, InterpolationMode interp, bool gauss_prefilter);
template void reduce(ImageGpu32sC4* src, ImageGpu32sC4* dst, InterpolationMode interp, bool gauss_prefilter);

template void reduce(ImageGpu32fC1* src, ImageGpu32fC1* dst, InterpolationMode interp, bool gauss_prefilter);
template void reduce(ImageGpu32fC2* src, ImageGpu32fC2* dst, InterpolationMode interp, bool gauss_prefilter);
template void reduce(ImageGpu32fC4* src, ImageGpu32fC4* dst, InterpolationMode interp, bool gauss_prefilter);


} // namespace cu
} // namespace imp

#endif // IMP_CU_REDUCE_IMPL_CU

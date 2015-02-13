#ifndef IMP_CU_GAUSS_IMPL_CU
#define IMP_CU_GAUSS_IMPL_CU

#include <imp/cuimgproc/cu_image_filter.cuh>

#include <cstdint>
#include <cuda_runtime.h>

#include <imp/core/types.hpp>
#include <imp/core/roi.hpp>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cucore/cu_utils.hpp>
#include <imp/cucore/cu_texture.cuh>



namespace imp {
namespace cu {

//-----------------------------------------------------------------------------
/** Perform a convolution with an gaussian smoothing kernel
 * @param dst          pointer to output image (linear memory)
 * @param stride       length of image row [pixels]
 * @param xoff         x-coordinate offset where to start the region [pixels]
 * @param yoff         y-coordinate offset where to start the region [pixels]
 * @param width        width of region [pixels]
 * @param height       height of region [pixels]
 * @param kernel_size  lenght of the smoothing kernel [pixels]
 * @param horizontal   defines the direction of convolution
 */
template<typename Pixel>
__global__ void k_gauss(Pixel* dst, const size_type stride,
                        const int xoff, const int yoff,
                        const int width, const int height,
                        Texture2D src_tex, int kernel_size, float c0,
                        float c1, bool horizontal=true)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  const size_type out_idx = y*stride+x;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    x += xoff;
    y += yoff;

    float sum = 0.0f;
    int half_kernel_elements = (kernel_size - 1) / 2;

    Pixel texel_c, texel;
    src_tex.fetch(texel_c, x, y);

    if (horizontal)
    {
      // convolve horizontally
      float g2 = c1 * c1;
      sum = c0 * texel_c;
      float sum_coeff = c0;

      for (int i = 1; i <= half_kernel_elements; i++)
      {
        c0 *= c1;
        c1 *= g2;
        int cur_x = max(0, min(width-1, x+i));
        src_tex.fetch(texel, cur_x, y);
        sum += c0 * texel;
        cur_x = max(0, min(width-1, x-i));
        src_tex.fetch(texel, cur_x, y);
        sum += c0 * texel;
        sum_coeff += 2.0f*c0;
      }
      dst[out_idx] = sum/sum_coeff;
    }
    else
    {
      // convolve vertically
      float g2 = c1 * c1;
      sum = c0 * texel_c;
      float sum_coeff = c0;

      for (int j = 1; j <= half_kernel_elements; j++)
      {
        c0 *= c1;
        c1 *= g2;
        float cur_y = max(0, min(height-1, y+j));
        src_tex.fetch(texel, x, cur_y);
        sum += c0 * texel;
        cur_y = max(0, min(height-1, y-j));
        src_tex.fetch(texel, x, cur_y);
        sum += c0 *  texel;
        sum_coeff += 2.0f*c0;
      }
      dst[out_idx] = sum/sum_coeff;
    }
  }
}


//-----------------------------------------------------------------------------
///** Perform a convolution with an gaussian smoothing kernel
// * @param dst          pointer to output image (linear memory)
// * @param stride       length of image row [pixels]
// * @param xoff         x-coordinate offset where to start the region [pixels]
// * @param yoff         y-coordinate offset where to start the region [pixels]
// * @param width        width of region [pixels]
// * @param height       height of region [pixels]
// * @param sigma        sigma of the smoothing kernel
// * @param kernel_size  lenght of the smoothing kernel [pixels]
// * @param horizontal   defines the direction of convolution
// */
//__global__ void cuFilterGaussZKernel_32f_C1(float* dst, float* src,
//                                            const int y,
//                                            const int width, const int depth,
//                                            const size_t stride, const size_t slice_stride,
//                                            float sigma, int kernel_size)
//{
//  int x = blockIdx.x*blockDim.x + threadIdx.x;
//  int z = blockIdx.y*blockDim.y + threadIdx.y;

//  if(x>=0 && z>= 0 && x<width && z<depth)
//  {
//    float sum = 0.0f;
//    int half_kernel_elements = (kernel_size - 1) / 2;

//    // convolve horizontally
//    float g0 = 1.0f / (sqrtf(2.0f * 3.141592653589793f) * sigma);
//    float g1 = exp(-0.5f / (sigma * sigma));
//    float g2 = g1 * g1;
//    sum = g0 * src[z*slice_stride + y*stride + x];
//    float sum_coeff = g0;
//    for (int i = 1; i <= half_kernel_elements; i++)
//    {
//      g0 *= g1;
//      g1 *= g2;
//      int cur_z = IUMAX(0, IUMIN(depth-1, z + i));
//      sum += g0 * src[cur_z*slice_stride + y*stride + x];
//      cur_z = IUMAX(0, IUMIN(depth-1, z - i));
//      sum += g0 * src[cur_z*slice_stride + y*stride + x];
//      sum_coeff += 2.0f*g0;
//    }
//    dst[z*slice_stride + y*stride + x] = sum/sum_coeff;
//  }
//}


//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
void filterGauss(ImageGpu<Pixel, pixel_type>* dst, ImageGpu<Pixel, pixel_type>* src,
                 float sigma, int kernel_size,
                 std::shared_ptr<ImageGpu<Pixel, pixel_type>> tmp_imp)
//                 cudaStream_t stream);
{
  if (kernel_size == 0)
    kernel_size = max(5, static_cast<int>(std::ceil(sigma*3)*2 + 1));
  if (kernel_size % 2 == 0)
    ++kernel_size;

  Roi2u roi = src->roi();

  // temporary variable for filtering (separabel kernel!)
  if (!tmp_imp || src->size() != tmp_imp->size());
  {
    tmp_imp.reset(new ImageGpu<Pixel, pixel_type>(roi.size()));
  }

 if (dst->roi().size() != roi.size())
    dst->setRoi(roi);

  // fragmentation
  Fragmentation<16,16> frag(roi);

  float c0 = 1.0f / (std::sqrt(2.0f * M_PI)*sigma);
  float c1 = std::exp(-0.5f / (sigma * sigma));

  // Convolve horizontally
  std::unique_ptr<Texture2D> src_tex =
      src->genTexture(false,(src->bitDepth()<32) ? cudaFilterModePoint
                                                 : cudaFilterModeLinear);
  k_gauss
      <<<
        frag.dimGrid, frag.dimBlock//, 0, stream
      >>> (tmp_imp->data(), tmp_imp->stride(),
           roi.x(), roi.y(), tmp_imp->width(), tmp_imp->height(),
           *src_tex, /*sigma, */kernel_size, c0, c1, false);

  // Convolve vertically
  src_tex = tmp_imp->genTexture(false,(tmp_imp->bitDepth()<32) ? cudaFilterModePoint
                                                               : cudaFilterModeLinear);
  k_gauss
      <<<
        frag.dimGrid, frag.dimBlock//, 0, stream
      >>> (dst->data(roi.x(), roi.y()), dst->stride(),
           roi.x(), roi.y(), roi.width(), roi.height(),
           *src_tex, /*sigma, */kernel_size, c0, c1, true);

  IMP_CUDA_CHECK();
}


//==============================================================================
//
// template instantiations for all our image types
//

template void filterGauss(ImageGpu8uC1* dst, ImageGpu8uC1* src, float sigma, int kernel_size, std::shared_ptr<ImageGpu8uC1> tmp_imp);
template void filterGauss(ImageGpu8uC2* dst, ImageGpu8uC2* src, float sigma, int kernel_size, std::shared_ptr<ImageGpu8uC2> tmp_imp);
template void filterGauss(ImageGpu8uC4* dst, ImageGpu8uC4* src, float sigma, int kernel_size, std::shared_ptr<ImageGpu8uC4> tmp_imp);

template void filterGauss(ImageGpu16uC1* dst, ImageGpu16uC1* src, float sigma, int kernel_size, std::shared_ptr<ImageGpu16uC1> tmp_imp);
template void filterGauss(ImageGpu16uC2* dst, ImageGpu16uC2* src, float sigma, int kernel_size, std::shared_ptr<ImageGpu16uC2> tmp_imp);
template void filterGauss(ImageGpu16uC4* dst, ImageGpu16uC4* src, float sigma, int kernel_size, std::shared_ptr<ImageGpu16uC4> tmp_imp);

template void filterGauss(ImageGpu32sC1* dst, ImageGpu32sC1* src, float sigma, int kernel_size, std::shared_ptr<ImageGpu32sC1> tmp_imp);
template void filterGauss(ImageGpu32sC2* dst, ImageGpu32sC2* src, float sigma, int kernel_size, std::shared_ptr<ImageGpu32sC2> tmp_imp);
template void filterGauss(ImageGpu32sC4* dst, ImageGpu32sC4* src, float sigma, int kernel_size, std::shared_ptr<ImageGpu32sC4> tmp_imp);

template void filterGauss(ImageGpu32fC1* dst, ImageGpu32fC1* src, float sigma, int kernel_size, std::shared_ptr<ImageGpu32fC1> tmp_imp);
template void filterGauss(ImageGpu32fC2* dst, ImageGpu32fC2* src, float sigma, int kernel_size, std::shared_ptr<ImageGpu32fC2> tmp_imp);
template void filterGauss(ImageGpu32fC4* dst, ImageGpu32fC4* src, float sigma, int kernel_size, std::shared_ptr<ImageGpu32fC4> tmp_imp);







/*
 *
 * KEEP THE 'OLD' STUFF AROUND IF WE WANT TO IMPLEMENT GAUSSIAN FILTER FOR VOLUMES (E.G.)
 *
 */

//// ----------------------------------------------------------------------------
//// wrapper: Gaussian filter; Volume; 32-bit; 1-channel
//void cuFilterGauss(const iu::VolumeGpu_32f_C1* src, iu::VolumeGpu_32f_C1* dst,
//                   float sigma, int kernel_size)
//{
//  if (kernel_size == 0)
//    kernel_size = max(5, (unsigned int)ceil(sigma*  3)*  2 + 1);
//  if (kernel_size%2 == 0)
//    ++kernel_size;

//  // temporary variable for filtering (separabel kernel!)
//  iu::VolumeGpu_32f_C1 tmpVol(src->size());


//  // fragmentation
//  unsigned int block_size = 16;
//  dim3 dimBlock(block_size, block_size);
//  dim3 dimGrid(iu::divUp(src->width(), dimBlock.x), iu::divUp(src->height(), dimBlock.y));

//  float c0 = 1.0f / (sqrtf(2.0f * 3.141592653589793f) * sigma);
//  float c1 = exp(-0.5f / (sigma * sigma));

//  // filter slices
//  for (int z=0; z<src->depth(); z++)
//  {
//    // temporary variable for filtering (separabed kernel!)
//    iu::ImageGpu_32f_C1 tmp(src->width(), src->height());

//    // textures
//    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
//    tex1_32f_C1__.filterMode = cudaFilterModeLinear;
//    tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
//    tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
//    tex1_32f_C1__.normalized = false;


//    // Convolve horizontally
//    cudaBindTexture2D(0, &tex1_32f_C1__, src->data(0,0,z), &channel_desc, src->width(), src->height(), src->pitch());
//    cuFilterGaussKernel_32f_C1 <<< dimGrid, dimBlock >>> (tmp.data(), tmp.stride(),
//                                                          0, 0, tmp.width(), tmp.height(),
//                                                          sigma, kernel_size, c0, c1, false);

//    // Convolve vertically
//    cudaBindTexture2D(0, &tex1_32f_C1__, tmp.data(), &channel_desc, tmp.width(), tmp.height(), tmp.pitch());
//    cuFilterGaussKernel_32f_C1 <<< dimGrid, dimBlock >>> (tmpVol.data(0,0,z), tmpVol.stride(),
//                                                          0, 0, tmpVol.width(), tmpVol.height(),
//                                                          sigma, kernel_size, c0, c1, true);

//    // unbind textures
//    cudaUnbindTexture(&tex1_32f_C1__);
//  }

//  cudaDeviceSynchronize();

//  dim3 dimGridZ(iu::divUp(src->width(), dimBlock.x), iu::divUp(src->depth(), dimBlock.y));

//  // filter slices
//  for (int y=0; y<src->height(); y++)
//  {
//    cuFilterGaussZKernel_32f_C1 <<< dimGridZ, dimBlock >>> (dst->data(), tmpVol.data(),
//                                                            y, dst->width(), dst->depth(),
//                                                            dst->stride(), dst->slice_stride(),
//                                                            sigma, kernel_size);
//  }


//  // error check
//  iu::checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__);
//}


} // namespace cu
} // namespace imp



#endif // IMP_CU_GAUSS_IMPL_CU

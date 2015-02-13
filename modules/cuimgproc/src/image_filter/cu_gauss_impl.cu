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

// ----------------------------------------------------------------------------
// kernel: Gaussian filter; 32-bit; 1-channel
/** Perform a convolution with an gaussian smoothing kernel
 * @param dst          pointer to output image (linear memory)
 * @param stride       length of image row [pixels]
 * @param xoff         x-coordinate offset where to start the region [pixels]
 * @param yoff         y-coordinate offset where to start the region [pixels]
 * @param width        width of region [pixels]
 * @param height       height of region [pixels]
 * @param sigma        sigma of the smoothing kernel
 * @param kernel_size  lenght of the smoothing kernel [pixels]
 * @param horizontal   defines the direction of convolution
 */
__global__ void cuFilterGaussKernel_32f_C1(float* dst, const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height,
                                           float sigma, int kernel_size, float c0,
                                           float c1, bool horizontal=true)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int oc = y*stride+x;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    float sum = 0.0f;
    int half_kernel_elements = (kernel_size - 1) / 2;

    if (horizontal)
    {
      // convolve horizontally
      float g2 = c1 * c1;
      sum = c0 * tex2D(tex1_32f_C1__, xx, yy);
      float sum_coeff = c0;
      for (int i = 1; i <= half_kernel_elements; i++)
      {
        c0 *= c1;
        c1 *= g2;
        float cur_xx = IUMAX(0.5f, IUMIN(width-0.5f, xx + i));
        sum += c0 * tex2D(tex1_32f_C1__, cur_xx, yy);
        cur_xx = IUMAX(0.5f, IUMIN(width-0.5f, xx-i));
        sum += c0 * tex2D(tex1_32f_C1__, cur_xx, yy);
        sum_coeff += 2.0f*c0;
      }
      dst[oc] = sum/sum_coeff;
    }
    else
    {
      // convolve vertically
      float g2 = c1 * c1;
      sum = c0 * tex2D(tex1_32f_C1__, xx, yy);
      float sum_coeff = c0;
      for (int j = 1; j <= half_kernel_elements; j++)
      {
        c0 *= c1;
        c1 *= g2;
        float cur_yy = IUMAX(0.5f, IUMIN(height-0.5f, yy+j));
        sum += c0 * tex2D(tex1_32f_C1__, xx, cur_yy);
        cur_yy = IUMAX(0.5f, IUMIN(height-0.5f, yy-j));
        sum += c0 *  tex2D(tex1_32f_C1__, xx, cur_yy);
        sum_coeff += 2.0f*c0;
      }
      dst[oc] = sum/sum_coeff;
    }
  }
}


// ----------------------------------------------------------------------------
// kernel: Gaussian filter; 32-bit; 1-channel
/** Perform a convolution with an gaussian smoothing kernel
 * @param dst          pointer to output image (linear memory)
 * @param stride       length of image row [pixels]
 * @param xoff         x-coordinate offset where to start the region [pixels]
 * @param yoff         y-coordinate offset where to start the region [pixels]
 * @param width        width of region [pixels]
 * @param height       height of region [pixels]
 * @param sigma        sigma of the smoothing kernel
 * @param kernel_size  lenght of the smoothing kernel [pixels]
 * @param horizontal   defines the direction of convolution
 */
__global__ void cuFilterGaussZKernel_32f_C1(float* dst, float* src,
                                            const int y,
                                            const int width, const int depth,
                                            const size_t stride, const size_t slice_stride,
                                            float sigma, int kernel_size)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int z = blockIdx.y*blockDim.y + threadIdx.y;

  if(x>=0 && z>= 0 && x<width && z<depth)
  {
    float sum = 0.0f;
    int half_kernel_elements = (kernel_size - 1) / 2;

    // convolve horizontally
    float g0 = 1.0f / (sqrtf(2.0f * 3.141592653589793f) * sigma);
    float g1 = exp(-0.5f / (sigma * sigma));
    float g2 = g1 * g1;
    sum = g0 * src[z*slice_stride + y*stride + x];
    float sum_coeff = g0;
    for (int i = 1; i <= half_kernel_elements; i++)
    {
      g0 *= g1;
      g1 *= g2;
      int cur_z = IUMAX(0, IUMIN(depth-1, z + i));
      sum += g0 * src[cur_z*slice_stride + y*stride + x];
      cur_z = IUMAX(0, IUMIN(depth-1, z - i));
      sum += g0 * src[cur_z*slice_stride + y*stride + x];
      sum_coeff += 2.0f*g0;
    }
    dst[z*slice_stride + y*stride + x] = sum/sum_coeff;
  }
}

// ----------------------------------------------------------------------------
// kernel: Gaussian filter; 32-bit; 4-channel
/** Perform a convolution with an gaussian smoothing kernel
 * @param dst          pointer to output image (linear memory)
 * @param stride       length of image row [pixels]
 * @param xoff         x-coordinate offset where to start the region [pixels]
 * @param yoff         y-coordinate offset where to start the region [pixels]
 * @param width        width of region [pixels]
 * @param height       height of region [pixels]
 * @param sigma        sigma of the smoothing kernel
 * @param kernel_size  lenght of the smoothing kernel [pixels]
 * @param horizontal   defines the direction of convolution
 */
__global__ void cuFilterGaussKernel_32f_C4(float4* dst, const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height,
                                           float sigma, int kernel_size,
                                           bool horizontal=true)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int oc = y*stride+x;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    int half_kernel_elements = (kernel_size - 1) / 2;

    if (horizontal)
    {
      // convolve horizontally
      float g0 = 1.0f / (sqrtf(2.0f * 3.141592653589793f) * sigma);
      float g1 = exp(-0.5f / (sigma * sigma));
      float g2 = g1 * g1;
      sum = g0 * tex2D(tex1_32f_C4__, xx, yy);
      float sum_coeff = g0;
      for (int i = 1; i <= half_kernel_elements; i++)
      {
        g0 *= g1;
        g1 *= g2;
        float cur_xx = IUMAX(0.5f, IUMIN(width-0.5f, xx + i));
        sum += g0 * tex2D(tex1_32f_C4__, cur_xx, yy);
        cur_xx = IUMAX(0.5f, IUMIN(width-0.5f, xx-i));
        sum += g0 * tex2D(tex1_32f_C4__, cur_xx, yy);
        sum_coeff += 2.0f*g0;
      }
      dst[oc] = sum/sum_coeff;
    }
    else
    {
      // convolve vertically
      float g0 = 1.0f / (sqrtf(2.0f * 3.141592653589793f) * sigma);
      float g1 = exp(-0.5f / (sigma * sigma));
      float g2 = g1 * g1;
      sum = g0 * tex2D(tex1_32f_C4__, xx, yy);
      float sum_coeff = g0;
      for (int j = 1; j <= half_kernel_elements; j++)
      {
        g0 *= g1;
        g1 *= g2;
        float cur_yy = IUMAX(0.5f, IUMIN(height-0.5f, yy+j));
        sum += g0 * tex2D(tex1_32f_C4__, xx, cur_yy);
        cur_yy = IUMAX(0.5f, IUMIN(height-0.5f, yy-j));
        sum += g0 *  tex2D(tex1_32f_C4__, xx, cur_yy);
        sum_coeff += 2.0f*g0;
      }
      dst[oc] = sum/sum_coeff;
    }
  }
}


// ----------------------------------------------------------------------------
// wrapper: Gaussian filter; 32-bit; 1-channel
void cuFilterGauss(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                   float sigma, int kernel_size, iu::ImageGpu_32f_C1* temp, cudaStream_t stream)
{
  if (kernel_size == 0)
    kernel_size = max(5, (unsigned int)ceil(sigma*  3)*  2 + 1);
  if (kernel_size%2 == 0)
    ++kernel_size;

  bool delete_local_temporary = false;

  // temporary variable for filtering (separabel kernel!)
  if (!temp)
  {
    temp = new iu::ImageGpu_32f_C1(src->size());
    delete_local_temporary = true;
  }


  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size*2, block_size/2);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  float c0 = 1.0f / (sqrtf(2.0f * 3.141592653589793f) * sigma);
  float c1 = exp(-0.5f / (sigma * sigma));

  // Convolve horizontally
  iu::bindTexture(tex1_32f_C1__, src);
  cuFilterGaussKernel_32f_C1 <<< dimGrid, dimBlock, 0, stream >>> (temp->data(roi.x, roi.y), temp->stride(),
                                                        roi.x, roi.y, temp->width(), temp->height(),
                                                        sigma, kernel_size, c0, c1, false);

  // Convolve vertically
  iu::bindTexture(tex1_32f_C1__, temp);
  cuFilterGaussKernel_32f_C1 <<< dimGrid, dimBlock, 0, stream >>> (dst->data(roi.x, roi.y), dst->stride(),
                                                        roi.x, roi.y, dst->width(), dst->height(),
                                                        sigma, kernel_size, c0, c1, true);


  if (delete_local_temporary)
    delete temp;

  // error check
  //iu::checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__);
}


// ----------------------------------------------------------------------------
// wrapper: Gaussian filter; Volume; 32-bit; 1-channel
void cuFilterGauss(const iu::VolumeGpu_32f_C1* src, iu::VolumeGpu_32f_C1* dst, float sigma, int kernel_size)
{
  if (kernel_size == 0)
    kernel_size = max(5, (unsigned int)ceil(sigma*  3)*  2 + 1);
  if (kernel_size%2 == 0)
    ++kernel_size;

  // temporary variable for filtering (separabel kernel!)
  iu::VolumeGpu_32f_C1 tmpVol(src->size());


  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(src->width(), dimBlock.x), iu::divUp(src->height(), dimBlock.y));

  float c0 = 1.0f / (sqrtf(2.0f * 3.141592653589793f) * sigma);
  float c1 = exp(-0.5f / (sigma * sigma));

  // filter slices
  for (int z=0; z<src->depth(); z++)
  {
    // temporary variable for filtering (separabed kernel!)
    iu::ImageGpu_32f_C1 tmp(src->width(), src->height());

    // textures
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
    tex1_32f_C1__.filterMode = cudaFilterModeLinear;
    tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
    tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
    tex1_32f_C1__.normalized = false;


    // Convolve horizontally
    cudaBindTexture2D(0, &tex1_32f_C1__, src->data(0,0,z), &channel_desc, src->width(), src->height(), src->pitch());
    cuFilterGaussKernel_32f_C1 <<< dimGrid, dimBlock >>> (tmp.data(), tmp.stride(),
                                                          0, 0, tmp.width(), tmp.height(),
                                                          sigma, kernel_size, c0, c1, false);

    // Convolve vertically
    cudaBindTexture2D(0, &tex1_32f_C1__, tmp.data(), &channel_desc, tmp.width(), tmp.height(), tmp.pitch());
    cuFilterGaussKernel_32f_C1 <<< dimGrid, dimBlock >>> (tmpVol.data(0,0,z), tmpVol.stride(),
                                                          0, 0, tmpVol.width(), tmpVol.height(),
                                                          sigma, kernel_size, c0, c1, true);

    // unbind textures
    cudaUnbindTexture(&tex1_32f_C1__);
  }

  cudaDeviceSynchronize();

  dim3 dimGridZ(iu::divUp(src->width(), dimBlock.x), iu::divUp(src->depth(), dimBlock.y));

  // filter slices
  for (int y=0; y<src->height(); y++)
  {
    cuFilterGaussZKernel_32f_C1 <<< dimGridZ, dimBlock >>> (dst->data(), tmpVol.data(),
                                                            y, dst->width(), dst->depth(),
                                                            dst->stride(), dst->slice_stride(),
                                                            sigma, kernel_size);
  }


  // error check
  iu::checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__);
}

// ----------------------------------------------------------------------------
// wrapper: Gaussian filter; 32-bit; 4-channel
void cuFilterGauss(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi, float sigma, int kernel_size)
{
  if (kernel_size == 0)
    kernel_size = max(5, (unsigned int)ceil(sigma*  3)*  2 + 1);
  if (kernel_size%2 == 0)
    ++kernel_size;

  // temporary variable for filtering (separabed kernel!)
  iu::ImageGpu_32f_C4 tmp(src->size());

  // textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
  tex1_32f_C4__.filterMode = cudaFilterModeLinear;
  tex1_32f_C4__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C4__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C4__.normalized = false;

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  // Convolve horizontally
  cudaBindTexture2D(0, &tex1_32f_C4__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());
  cuFilterGaussKernel_32f_C4 <<< dimGrid, dimBlock >>> (tmp.data(roi.x, roi.y), tmp.stride(),
                                                        roi.x, roi.y, tmp.width(), tmp.height(),
                                                        sigma, kernel_size, false);
  cudaUnbindTexture(tex1_32f_C4__);

  // Convolve vertically
  cudaBindTexture2D(0, &tex1_32f_C4__, tmp.data(), &channel_desc, tmp.width(), tmp.height(), tmp.pitch());
  cuFilterGaussKernel_32f_C4 <<< dimGrid, dimBlock >>> (dst->data(roi.x, roi.y), dst->stride(),
                                                        roi.x, roi.y, dst->width(), dst->height(),
                                                        sigma, kernel_size, true);
  cudaUnbindTexture(&tex1_32f_C4__);

  // error check
  iu::checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__);
}

} // namespace cu
} // namespace imp



#endif IMP_CU_GAUSS_IMPL_CU

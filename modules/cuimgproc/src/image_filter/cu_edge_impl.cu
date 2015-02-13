#ifndef IMP_CU_EDGE_IMPL_CU
#define IMP_CU_EDGE_IMPL_CU

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


// -- C1 -> C2 ---------------------------------------------------------------
// kernel: edge filter; 32-bit; 1-channel
__global__ void  cuFilterEdgeKernel_32f_C1(float2* dst, const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    dst[y*stride+x] = make_float2(tex2D(tex1_32f_C1__, xx+1.0f, yy) - tex2D(tex1_32f_C1__, xx, yy),
                                  tex2D(tex1_32f_C1__, xx, yy+1.0f) - tex2D(tex1_32f_C1__, xx, yy) );
  }
}

// ----------------------------------------------------------------------------
// wrapper: edge filter
void cuFilterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  tex1_32f_C1__.filterMode = cudaFilterModeLinear;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(),
                    src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  cuFilterEdgeKernel_32f_C1<<<dimGrid, dimBlock>>>(dst->data(roi.x, roi.y), dst->stride(),
                                                   roi.x, roi.y, roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C1__);

  // error check
  iu::checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__);
}


// -- C1 -> C4 - Eval --------------------------------------------------------
// kernel: edge filter + evaluation; 32-bit; 1-channel
__global__ void  cuFilterEdgeKernel_32f_C1(float4* dst, float alpha, float beta, float minval,
                                           const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    dst[y*stride+x] = make_float4(max(minval, exp(-alpha*pow(abs(tex2D(tex1_32f_C1__, xx+1.0f, yy)      - tex2D(tex1_32f_C1__, xx, yy)), beta))),
                                  max(minval, exp(-alpha*pow(abs(tex2D(tex1_32f_C1__, xx, yy+1.0f)      - tex2D(tex1_32f_C1__, xx, yy)), beta))),
                                  max(minval, exp(-alpha*pow(abs(tex2D(tex1_32f_C1__, xx+1.0f, yy+1.0f) - tex2D(tex1_32f_C1__, xx, yy)), beta))),
                                  max(minval, exp(-alpha*pow(abs(tex2D(tex1_32f_C1__, xx+1.0f, yy-1.0f) - tex2D(tex1_32f_C1__, xx, yy)), beta))) );
  }
}

// ----------------------------------------------------------------------------
// wrapper: edge filter  + evaluation
void cuFilterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C4* dst,
                      const IuRect& roi, float alpha, float beta, float minval)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  tex1_32f_C1__.filterMode = cudaFilterModeLinear;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(),
                    src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  cuFilterEdgeKernel_32f_C1<<<dimGrid, dimBlock>>>(dst->data(roi.x, roi.y), alpha, beta,
                                                   minval, dst->stride(), roi.x, roi.y,
                                                   roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C1__);

  // error check
  iu::checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__);
}


// -- C1 -> C2 - Eval --------------------------------------------------------
// kernel: edge filter + evaluation; 32-bit; 1-channel
__global__ void  cuFilterEdgeKernel_32f_C1(float2* dst, float alpha, float beta, float minval,
                                           const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    dst[y*stride+x] = make_float2(max(minval, exp(-alpha*pow(abs(tex2D(tex1_32f_C1__, xx+1.0f, yy) - tex2D(tex1_32f_C1__, xx, yy)), beta))),
                                  max(minval, exp(-alpha*pow(abs(tex2D(tex1_32f_C1__, xx, yy+1.0f) - tex2D(tex1_32f_C1__, xx, yy)), beta))) );
  }
}

// ----------------------------------------------------------------------------
// wrapper: edge filter  + evaluation
void cuFilterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C2* dst,
                      const IuRect& roi, float alpha, float beta, float minval)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  tex1_32f_C1__.filterMode = cudaFilterModeLinear;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(),
                    src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  cuFilterEdgeKernel_32f_C1<<<dimGrid, dimBlock>>>(dst->data(roi.x, roi.y), alpha, beta,
                                                   minval, dst->stride(), roi.x, roi.y,
                                                   roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C1__);

  // error check
  iu::checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__);
}


// -- C1 -> C1 - Eval --------------------------------------------------------
// kernel: edge filter + evaluation; 32-bit; 1-channel
__global__ void  cuFilterEdgeKernel_32f_C1(float* dst, float alpha, float beta, float minval,
                                           const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    float2 grad = make_float2(tex2D(tex1_32f_C1__, xx+1.0f, yy) - tex2D(tex1_32f_C1__, xx, yy),
                              tex2D(tex1_32f_C1__, xx, yy+1.0f) - tex2D(tex1_32f_C1__, xx, yy) );
    dst[y*stride+x] = max(minval, exp(-alpha*pow(length(grad), beta)));
  }
}

// ----------------------------------------------------------------------------
// wrapper: edge filter  + evaluation
void cuFilterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                      float alpha, float beta, float minval)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  tex1_32f_C1__.filterMode = cudaFilterModeLinear;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(),
                    src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  cuFilterEdgeKernel_32f_C1 <<< dimGrid, dimBlock >>> (dst->data(roi.x, roi.y), alpha, beta,
                                                       minval, dst->stride(), roi.x, roi.y,
                                                       roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C1__);

  // error check
  iu::checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__);
}


// -- RGB -> C1 - Eval --------------------------------------------------------
// kernel: edge filter + evaluation; 32-bit; 4-channel
__global__ void  cuFilterEdgeKernel_32f_C4(float* dst, float alpha, float beta, float minval,
                                           const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    float4 gradx = tex2D(tex1_32f_C4__, xx+1.0f, yy) - tex2D(tex1_32f_C4__, xx, yy);
    float4 grady = tex2D(tex1_32f_C4__, xx, yy+1.0f) - tex2D(tex1_32f_C4__, xx, yy);
    float3 grad;
    grad.x = sqrtf(gradx.x*gradx.x + grady.x*grady.x);
    grad.y = sqrtf(gradx.y*gradx.y + grady.y*grady.y);
    grad.z = sqrtf(gradx.z*gradx.z + grady.z*grady.z);
    dst[y*stride+x] = max(minval, exp(-alpha*pow((grad.x+grad.y+grad.z)/3.0f, beta)));
  }
}

// ----------------------------------------------------------------------------
// wrapper: edge filter  + evaluation
void cuFilterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                      float alpha, float beta, float minval)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
  tex1_32f_C4__.filterMode = cudaFilterModeLinear;
  tex1_32f_C4__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C4__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C4__.normalized = false;
  cudaBindTexture2D(0, &tex1_32f_C4__, src->data(), &channel_desc, src->width(),
                    src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  cuFilterEdgeKernel_32f_C4 <<< dimGrid, dimBlock >>> (dst->data(roi.x, roi.y), alpha, beta,
                                                       minval, dst->stride(), roi.x, roi.y,
                                                       roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C4__);

  // error check
  iu::checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__);
}

// -- RGB -> C2 - Eval --------------------------------------------------------
// kernel: edge filter + evaluation; 32-bit; 4-channel
__global__ void  cuFilterEdgeKernel_32f_C4(float2* dst, float alpha, float beta, float minval,
                                           const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    float4 gradx = tex2D(tex1_32f_C4__, xx+1.0f, yy) - tex2D(tex1_32f_C4__, xx, yy);
    float4 grady = tex2D(tex1_32f_C4__, xx, yy+1.0f) - tex2D(tex1_32f_C4__, xx, yy);
    float valx = (abs(gradx.x) + abs(gradx.y) + abs(gradx.z))/3.0f;
    float valy = (abs(grady.x) + abs(grady.y) + abs(grady.z))/3.0f;

    dst[y*stride+x] = make_float2(max(minval, exp(-alpha*pow(valx, beta))),
                                  max(minval, exp(-alpha*pow(valy, beta)))  );
  }
}

// ----------------------------------------------------------------------------
// wrapper: edge filter  + evaluation
void cuFilterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi,
                      float alpha, float beta, float minval)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
  tex1_32f_C4__.filterMode = cudaFilterModeLinear;
  tex1_32f_C4__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C4__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C4__.normalized = false;
  cudaBindTexture2D(0, &tex1_32f_C4__, src->data(), &channel_desc, src->width(),
                    src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  cuFilterEdgeKernel_32f_C4 <<< dimGrid, dimBlock >>> (dst->data(roi.x, roi.y), alpha, beta,
                                                       minval, dst->stride(), roi.x, roi.y,
                                                       roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C4__);

  // error check
  iu::checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__);
}


// -- RGB -> C4 - Eval --------------------------------------------------------
// kernel: edge filter + evaluation; 32-bit; 4-channel
__global__ void  cuFilterEdgeKernel_32f_C4(float4* dst, float alpha, float beta, float minval,
                                           const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    float4 grad = tex2D(tex1_32f_C4__, xx+1.0f, yy) - tex2D(tex1_32f_C4__, xx, yy);
    float valx = (abs(grad.x) + abs(grad.y) + abs(grad.z))/3.0f;
    grad = tex2D(tex1_32f_C4__, xx, yy+1.0f) - tex2D(tex1_32f_C4__, xx, yy);
    float valy = (abs(grad.x) + abs(grad.y) + abs(grad.z))/3.0f;
    grad = tex2D(tex1_32f_C4__, xx+1.0f, yy+1.0f) - tex2D(tex1_32f_C4__, xx, yy);
    float valxy = (abs(grad.x) + abs(grad.y) + abs(grad.z))/3.0f;
    grad = tex2D(tex1_32f_C4__, xx+1.0f, yy-1.0f) - tex2D(tex1_32f_C4__, xx, yy);
    float valxy2 = (abs(grad.x) + abs(grad.y) + abs(grad.z))/3.0f;

    dst[y*stride+x] = make_float4(max(minval, exp(-alpha*pow(valx, beta))),
                                  max(minval, exp(-alpha*pow(valy, beta))),
                                  max(minval, exp(-alpha*pow(valxy, beta))),
                                  max(minval, exp(-alpha*pow(valxy2, beta))) );
  }
}

// ----------------------------------------------------------------------------
// wrapper: edge filter  + evaluation
void cuFilterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi,
                      float alpha, float beta, float minval)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
  tex1_32f_C4__.filterMode = cudaFilterModeLinear;
  tex1_32f_C4__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C4__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C4__.normalized = false;
  cudaBindTexture2D(0, &tex1_32f_C4__, src->data(), &channel_desc, src->width(),
                    src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  cuFilterEdgeKernel_32f_C4 <<< dimGrid, dimBlock >>> (dst->data(roi.x, roi.y), alpha, beta,
                                                       minval, dst->stride(),
                                                       roi.x, roi.y,
                                                       roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C4__);

  // error check
  iu::checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__);
}

/* *************************************************************************** */

//-----------------------------------------------------------------------------
// wrapper: cubic bspline coefficients prefilter.
void cuCubicBSplinePrefilter_32f_C1I(iu::ImageGpu_32f_C1 *input)
{
  const unsigned int block_size = 64;
  const unsigned int width  = input->width();
  const unsigned int height = input->height();

  dim3 dimBlockX(block_size,1,1);
  dim3 dimGridX(iu::divUp(height, block_size),1,1);
  cuSamplesToCoefficients2DX<float> <<< dimGridX, dimBlockX >>> (input->data(), width, height, input->stride());

  dim3 dimBlockY(block_size,1,1);
  dim3 dimGridY(iu::divUp(width, block_size),1,1);
  cuSamplesToCoefficients2DY<float> <<< dimGridY, dimBlockY >>> (input->data(), width, height, input->stride());

  iu::checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__);
}


} // namespace cu
} // namespace imp



#endif IMP_CU_EDGE_IMPL_CU

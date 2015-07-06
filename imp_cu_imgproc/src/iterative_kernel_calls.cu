#include <imp/cu_imgproc/iterative_kernel_calls.cuh>

#include <iostream>

#include <cuda_runtime.h>

#include <imp/core/pixel.hpp>
#include <imp/cu_core/cu_texture.cuh>

namespace imp {
namespace cu {

//-----------------------------------------------------------------------------
__global__ void kernelCall(Pixel32fC1* d_u, size_t stride_u,
                           imp::cu::Texture2D in_tex,
                           size_t width, size_t height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x<width && y<height)
  {
    float val = tex2DFetch<float>(in_tex, x, y);
    d_u[y*stride_u + x] = val;
  }
}

//#############################################################################

//-----------------------------------------------------------------------------
IterativeKernelCalls::IterativeKernelCalls()
  : in_tex_(nullptr)
{
}

//-----------------------------------------------------------------------------
IterativeKernelCalls::~IterativeKernelCalls()
{

}

//-----------------------------------------------------------------------------
void IterativeKernelCalls::init(const Size2u& size)
{
  size_ = size;
  imp::cu::Fragmentation<> fragmentation(size);

  // setup internal memory
  out_.reset(new ImageGpu32fC1(size));
  IMP_CUDA_CHECK();

  // setup textures
  in_tex_ = in_->genTexture(false, cudaFilterModeLinear, cudaAddressModeClamp, cudaReadModeElementType);
  IMP_CUDA_CHECK();

  // init internal vars
  kernelCall
      <<< fragmentation.dimGrid, fragmentation.dimBlock
      >>> (out_->data(), out_->stride(), *in_tex_, size_.width(), size_.height());
  IMP_CUDA_CHECK();


  unrelated_.reset(new ImageGpu32fC1(size_));
  unrelated_->setValue(0);


  IMP_CUDA_CHECK();
}

//-----------------------------------------------------------------------------
void IterativeKernelCalls::run(const imp::cu::ImageGpu32fC1::Ptr& dst,
                               const imp::cu::ImageGpu32fC1::Ptr& src,
                               bool break_things)
{
  in_ = src;
  this->init(in_->size());
  imp::cu::Fragmentation<> fragmentation(in_->size());

  if (break_things)
  {
    this->breakThings();
  }

  kernelCall
      <<< fragmentation.dimGrid, fragmentation.dimBlock
      >>> (out_->data(), out_->stride(), *in_tex_, size_.width(), size_.height());
  IMP_CUDA_CHECK();

  out_->copyTo(*dst);
  IMP_CUDA_CHECK();
}

//-----------------------------------------------------------------------------
void IterativeKernelCalls::breakThings()
{
#if 0
  imp::cu::Texture2D::Ptr unrelated_tex = unrelated_->genTexture(
        false, cudaFilterModeLinear, cudaAddressModeClamp, cudaReadModeElementType);
#endif

#if 1
  cudaTextureObject_t tex_object;

  cudaResourceDesc tex_res;
  std::memset(&tex_res, 0, sizeof(tex_res));
  tex_res.resType = cudaResourceTypePitch2D;
  tex_res.res.pitch2D.width = unrelated_->width();
  tex_res.res.pitch2D.height = unrelated_->height();
  tex_res.res.pitch2D.pitchInBytes = unrelated_->pitch();
  tex_res.res.pitch2D.devPtr = (void*)unrelated_->data();
  tex_res.res.pitch2D.desc = unrelated_->channelFormatDesc();

  cudaTextureDesc tex_desc;
  std::memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.normalizedCoords = 0;
  tex_desc.filterMode = cudaFilterModeLinear;
  tex_desc.addressMode[0] = cudaAddressModeClamp;
  tex_desc.addressMode[1] = cudaAddressModeClamp;
  tex_desc.readMode = cudaReadModeElementType;

  cudaError_t err = cudaCreateTextureObject(&tex_object, &tex_res, &tex_desc, 0);
  if  (err != ::cudaSuccess)
  {
    throw imp::cu::Exception("Failed to create texture object", err,
                             __FILE__, __FUNCTION__, __LINE__);
  }
#endif

  IMP_CUDA_CHECK();
}


} // namespace cu
} // namespace imp

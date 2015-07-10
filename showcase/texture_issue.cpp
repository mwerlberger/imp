#include <stdlib.h>
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <imp/core/roi.hpp>
#include <imp/core/image_raw.hpp>
#include <imp/cu_core/cu_image_gpu.cuh>

#include "iterative_kernel_calls.cuh"

//-----------------------------------------------------------------------------
template <typename Pixel, int memaddr_align=32>
static Pixel* alignedAllocCpu(const size_t width, const size_t height,
                              size_t* pitch, bool init_with_zeros=false)
{
  assert(width>0 && height>0);
  // restrict the memory address alignment to be in the interval ]0,128] and
  // of power-of-two using the 'complement and compare' method
  assert((memaddr_align != 0) && memaddr_align <= 128 &&
         ((memaddr_align & (~memaddr_align + 1)) == memaddr_align));

  // check if the width allows a correct alignment of every row, otherwise add padding
  const size_t width_bytes = width * sizeof(Pixel);
  // bytes % memaddr_align = 0 for bytes=n*memaddr_align is the reason for
  // the decrement in the following compution:
  const size_t bytes_to_add = (memaddr_align-1) - ((width_bytes-1) % memaddr_align);
  const size_t pitched_width = width + bytes_to_add/sizeof(Pixel);
  *pitch = width_bytes + bytes_to_add;

  size_t num_elements = pitched_width*height;
  size_t memory_size = sizeof(Pixel)*num_elements;
  Pixel* p_data_aligned;
  int ret = posix_memalign((void**)&p_data_aligned, memaddr_align, memory_size);
  if (p_data_aligned == nullptr || ret != 0)
  {
    throw std::bad_alloc();
  }
  return p_data_aligned;
}

//-----------------------------------------------------------------------------
template <typename Pixel>
static Pixel* alignedAllocGpu(const size_t width, const size_t height,
                              size_t* pitch)
{
  if (width == 0 || height == 0)
  {
    throw imp::cu::Exception("Failed to allocate memory: width or height is zero");
  }

  size_t width_bytes = width * sizeof(Pixel);

  size_t intern_pitch;
  Pixel* p_data = nullptr;
  cudaError_t cu_err = cudaMallocPitch((void **)&p_data, &intern_pitch,
                                       width_bytes, (size_t)height);

  *pitch = intern_pitch;

  if (cu_err != cudaSuccess)
  {
    throw std::bad_alloc();
  }
  return p_data;
}

//=============================================================================
int main(int argc, char** argv)
{
  try
  {
    //std::cout << "usage: texture_issue [break_things_flag]";

    size_t width = 250;
    size_t height = 250;
    size_t x_off = width/3;
    size_t y_off = height/3;
    size_t roi_width = width/3;
    size_t roi_height = height/3;

    // alloc cpu memory (aligned)
    size_t pitch_cpu, stride_cpu;
    float* image_cpu = alignedAllocCpu<float,32>(width, height, &pitch_cpu);
    float* result_cpu = alignedAllocCpu<float,32>(width, height, &pitch_cpu);
    stride_cpu = pitch_cpu/sizeof(float);
    std::fill(image_cpu, image_cpu+(stride_cpu*height), 0.f);

    for(size_t y=y_off; y<y_off+roi_height; ++y)
    {
      for(size_t x=x_off; x<x_off+roi_width; ++x)
      {
        image_cpu[y*stride_cpu+x] = 1.0f;
      }
    }

    // alloc gpu memory
    size_t pitch_gpu, stride_gpu;
    float* image_gpu = alignedAllocGpu<float>(width,height,&pitch_gpu);
    float* result_gpu = alignedAllocGpu<float>(width,height,&pitch_gpu);
    stride_gpu = pitch_gpu / sizeof(float);

    imp::cu::ImageGpu32fC1::Ptr cu_image = std::make_shared<imp::cu::ImageGpu32fC1>(width, height);//(cols, rows);
    imp::cu::ImageGpu32fC1::Ptr cu_result = std::make_shared<imp::cu::ImageGpu32fC1>(width, height);//(cols, rows);

    cudaMemcpy2D(cu_image->data(), cu_image->pitch(), image_cpu, pitch_cpu, width*sizeof(float), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(image_gpu, pitch_gpu, image_cpu, pitch_cpu, width*sizeof(float), height, cudaMemcpyHostToDevice);
    IMP_CUDA_CHECK();

    //
    //
    imp::cu::IterativeKernelCalls ikc;
    bool break_things = (argc>1) ? true : false;
    ikc.run(cu_result, cu_image, break_things);
    IMP_CUDA_CHECK();
    //
    cudaMemcpy2D(result_gpu, pitch_gpu, image_gpu, pitch_gpu, width*sizeof(float), height, cudaMemcpyDeviceToDevice);
    //
    //
//    cudaMemcpy2D(result_cpu, pitch_cpu, cu_result->data(), cu_result->pitch(), cu_result->rowBytes(), cu_result->height(), cudaMemcpyDeviceToHost);
    cudaMemcpy2D(result_cpu, pitch_cpu, result_gpu, pitch_gpu, width*sizeof(float), height, cudaMemcpyDeviceToHost);
    IMP_CUDA_CHECK();

    float in_sum = 0.f;
    float out_sum = 0.f;
    for(size_t y=0; y<height; ++y)
    {
      for(size_t x=0; x<width; ++x)
      {
        in_sum += image_cpu[y*stride_cpu+x];
        out_sum += result_cpu[y*stride_cpu+x];
      }
    }
    std::cout << "in_sum: " << in_sum << "; out_sum: " << out_sum << std::endl;

    cv::Mat vis_in(height, width, CV_32FC1, image_cpu, pitch_cpu);
    cv::Mat vis_out(height, width, CV_32FC1, result_cpu, pitch_cpu);

    cv::imshow("input", vis_in);
    cv::imshow("output", vis_out);
    cv::waitKey();

    (void)stride_gpu;

  }
  catch (std::exception& e)
  {
    std::cout << "[exception] " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;

}

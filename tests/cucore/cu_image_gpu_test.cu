#include <assert.h>
#include <cstdint>
#include <iostream>

#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <imp/core/roi.hpp>
#include <imp/core/image_raw.hpp>
#include <imp/core/image_cv.hpp>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cucore/cu_utils.hpp>

#include "default_msg.h"


__global__ void testKernel(int val, const imp::Pixel8uC1 value)
{
    printf("[%d, %d]:\t\tValue is:%d\n",\
           blockIdx.y*gridDim.x+blockIdx.x,\
           threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
           int(value.c[0]));
}

template<typename Pixel>
__global__ void testSizeKernel(Pixel* dst, size_t stride, size_t width, size_t height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  printf("[%d, %d], dst=%p, stride=%lu, width=%lu, height=%lu\n", x, y, dst, stride, width, height);
  dst[y*stride+x].c[0] = 255;
  dst[y*stride+x].c[1] = 0;
  dst[y*stride+x].c[2] = 0;

}
__global__ void blaKernel(imp::Pixel8uC1* dst, size_t stride, /*const Pixel value*/std::uint8_t value,
                          size_t width, size_t height)
{
//  printf("[%d, %d]:\t\tValue is:%d\n",\
//          blockIdx.y*gridDim.x+blockIdx.x,\
//          threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
//          (int)value.c[0]);
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;


  if (x>=0 && y>=0 && x<width && y<height)
  {
    int c=y*stride+x;
//    printf("x=%d, y=%d, w=%d, h=%d, s=%d, dst=%p, value=[%d,%d,%d], c=%d\n",
//           x, y, width, height, stride, dst, value, value, value, c);
    dst[c].x = value;
  }
}


template<class Pixel>
__global__ void blaKernel(Pixel* dst, size_t stride, /*const Pixel value*/int value,
                          size_t width, size_t height)
{
//  printf("[%d, %d]:\t\tValue is:%d\n",\
//          blockIdx.y*gridDim.x+blockIdx.x,\
//          threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
//          (int)value.c[0]);
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;


  if (x>=0 && y>=0 && x<width && y<height)
  {
    int c=y*stride+x;
//    printf("x=%d, y=%d, w=%d, h=%d, s=%d, dst=%p, value=[%d,%d,%d], c=%d\n",
//           x, y, width, height, stride, dst, value, value, value, c);
    dst[c].x = value;
    dst[c].y = 0;
    dst[c].z = 0;
  }
}



int main(int argc, char** argv)
{
  try
  {

////    int devID;
////    cudaDeviceProp props;

////    // This will pick the best possible CUDA capable device
////    devID = findCudaDevice(argc, (const char **)argv);

////    //Get GPU information
////    checkCudaErrors(cudaGetDevice(&devID));
////    checkCudaErrors(cudaGetDeviceProperties(&props, devID));
////    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
////           devID, props.name, props.major, props.minor);

    printf("printf() is called. Output:\n\n");

    //Kernel configuration, where a two-dimensional grid and
    //three-dimensional blocks are configured.
    dim3 dimGrid1(2, 2);
    dim3 dimBlock1(2, 2, 2);
    testKernel<<<dimGrid1, dimBlock1>>>(10, imp::Pixel8uC1(123));
    cudaDeviceSynchronize();

//    //
//    //
//    //

    imp::ImageCv8uC1 h1_lena_8uC1(cv::imread("/home/mwerlberger/data/std/Lena.tiff",
                                             CV_LOAD_IMAGE_GRAYSCALE),
                                  imp::PixelOrder::gray);

    imp::ImageCv8uC3 h1_lena_8uC3(cv::imread("/home/mwerlberger/data/std/Lena.tiff",
                                             CV_LOAD_IMAGE_COLOR),
                                  imp::PixelOrder::bgr);

//    imp::ImageRaw8uC1 h1_lena_8uC1(
//          reinterpret_cast<imp::ImageRaw8uC1::pixel_container_t>(lena.data),
//          lena.cols, lena.rows, lena.step, true);

    // copy host->device->device->host
    imp::cu::ImageGpu8uC1 d1_lena_8uC1(h1_lena_8uC1);
    imp::cu::ImageGpu8uC1 d2_lena_8uC1(d1_lena_8uC1);
    imp::ImageCv8uC1 h2_lena_8uC1(d2_lena_8uC1);

    // same for 3-channel image
    imp::cu::ImageGpu8uC3 d1_lena_8uC3(h1_lena_8uC3);
    imp::cu::ImageGpu8uC3 d2_lena_8uC3(d1_lena_8uC3);
    imp::ImageCv8uC3 h2_lena_8uC3(d2_lena_8uC3);


    //--------------------------------------------------------------------------
    // image size kernel check
    {
      imp::Pixel8uC1 blaC1(123);
      imp::Pixel8uC3 blaC3(123,234,101);
      printf("C1: %lu; C3: %lu\n", sizeof(blaC1), sizeof(blaC3));

      imp::Size2u dd_sz(513, 512);
      imp::cu::ImageGpu8uC3 dd0_8uC3(dd_sz);

      printf("stride: %lu, pitch: %lu\n", dd0_8uC3.stride(), dd0_8uC3.pitch());

      return EXIT_SUCCESS;

      const unsigned int block_size=16;
      dim3 dimBlock(block_size, block_size);
      dim3 dimGrid(imp::cu::divUp(dd0_8uC3.width(), dimBlock.x),
                   imp::cu::divUp(dd0_8uC3.height(), dimBlock.y));
      testSizeKernel
          <<< dimGrid, dimBlock >>> (dd0_8uC3.data(), dd0_8uC3.stride(),
                                     dd0_8uC3.width(), dd0_8uC3.height());
      cudaThreadSynchronize();

      imp::ImageCv8uC3 hh0_8uC3(dd0_8uC3);
      cv::imshow("test - 8uC3", hh0_8uC3.cvMat());
      cv::waitKey();
      return EXIT_SUCCESS;
    }

    // set-value test:
    imp::Size2u sz(513, 512);
    printf("sz: %dx%d", sz.width(), sz.height());

    imp::cu::ImageGpu8uC1 d0_8uC1(sz);
    d0_8uC1.setValue(imp::Pixel8uC1(128));
    imp::ImageCv8uC1 h0_8uC1(d0_8uC1);

    imp::cu::ImageGpu8uC3 d1_8uC3(sz);
    printf("d1_8uC3: %dx%d", d1_8uC3.width(), d1_8uC3.height());
    //d1_8uC3.setValue(imp::Pixel8uC3(255, 0, 0));

    // fragmentation
    const unsigned int block_size = 16;
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid(imp::cu::divUp(d1_8uC3.width(), dimBlock.x),
                 imp::cu::divUp(d1_8uC3.height(), dimBlock.y));
    // todo add roi to kernel!
    printf("w/h: %dx%d\n", d1_8uC3.width(), d1_8uC3.height());
    blaKernel
        <<< dimGrid, dimBlock >>> (d1_8uC3.data(), d1_8uC3.stride(), /*imp::Pixel8uC3(255, 0, 0)*/255,
                                   d1_8uC3.width(), d1_8uC3.height());
    cudaDeviceSynchronize();
    imp::ImageCv8uC3 h1_8uC3(d1_8uC3);


    std::cout << "h0_8uC1: " << h0_8uC1 << std::endl;
    std::cout << "d0_8uC1: " << d0_8uC1 << std::endl;
//    std::cout << "h1_8uC3: " << h1_8uC3 << std::endl;
    std::cout << "d1_8uC3: " << d1_8uC3 << std::endl;

    std::cout << "h1_lena_8uC1: " << h1_lena_8uC1 << std::endl;
    std::cout << "d1_lena_8uC1: " << d1_lena_8uC1 << std::endl;
    std::cout << "h2_lena_8uC1: " << h2_lena_8uC1 << std::endl;
    std::cout << "d2_lena_8uC1: " << d2_lena_8uC1 << std::endl;

    std::cout << "h1_lena_8uC3: " << h1_lena_8uC3 << std::endl;
    std::cout << "d1_lena_8uC3: " << d1_lena_8uC3 << std::endl;
    std::cout << "h2_lena_8uC3: " << h2_lena_8uC3 << std::endl;
    std::cout << "d2_lena_8uC3: " << d2_lena_8uC3 << std::endl;

    cv::imshow("setValue - 8uC1", h0_8uC1.cvMat());
    cv::imshow("setValue - 8uC3 - is it blue??", h1_8uC3.cvMat());

    cv::imshow("lena input", h1_lena_8uC1.cvMat());
    cv::imshow("lena copied around", h2_lena_8uC1.cvMat());

    cv::imshow("COLOR lena input", h1_lena_8uC3.cvMat());
    cv::imshow("COLOR lena copied around", h2_lena_8uC3.cvMat());

    cv::waitKey();
  }
  catch (std::exception& e)
  {
    std::cout << "[exception] " << e.what() << std::endl;
    assert(false);
  }

  std::cout << imp::ok_msg << std::endl;

  return EXIT_SUCCESS;

}

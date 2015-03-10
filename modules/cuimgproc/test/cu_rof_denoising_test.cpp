#include <assert.h>
#include <cstdint>
#include <iostream>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <imp/core/roi.hpp>
#include <imp/core/image_raw.hpp>
#include <imp/core/image_cv.hpp>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cuimgproc/cu_rof_denoising.cuh>
#include <imp/io/opencv_bridge.hpp>

int main(int /*argc*/, char** /*argv*/)
{
  try
  {
    // ROF denoising 8uC1
    {
      std::shared_ptr<imp::cu::ImageGpu8uC1> d1_lena_8uC1 =
          imp::cu::ocvBridgeLoad<imp::Pixel8uC1,imp::PixelType::i8uC1>(
            "/home/mwerlberger/data/std/Lena.tiff", imp::PixelOrder::gray);
      std::shared_ptr<imp::cu::ImageGpu8uC1> d_lena_denoised_8uC1(
            new imp::cu::ImageGpu8uC1(*d1_lena_8uC1));

      imp::cu::RofDenoising8uC1 rof;
      std::cout << "\n" << rof << std::endl << std::endl;
      rof.denoise(d_lena_denoised_8uC1, d1_lena_8uC1);

      // show results
      imp::cu::ocvBridgeShow("lena input 8u", *d1_lena_8uC1);
      imp::cu::ocvBridgeShow("lena denoised 8u", *d_lena_denoised_8uC1);
    }

//    imp::ocvBridgeSave("test.png", h_lena_denoised_8uC1);

    // ROF denoising 32fC1
    {
      std::shared_ptr<imp::cu::ImageGpu32fC1> d1_lena_32fC1 =
          imp::cu::ocvBridgeLoad<imp::Pixel32fC1,imp::PixelType::i32fC1>(
            "/home/mwerlberger/data/std/Lena.tiff", imp::PixelOrder::gray);
      std::shared_ptr<imp::cu::ImageGpu32fC1> d_lena_denoised_32fC1(
            new imp::cu::ImageGpu32fC1(*d1_lena_32fC1));

      imp::cu::RofDenoising32fC1 rof_32fC1;
      rof_32fC1.denoise(d_lena_denoised_32fC1, d1_lena_32fC1);

      imp::cu::ocvBridgeShow("lena input 32f", *d1_lena_32fC1);
      imp::cu::ocvBridgeShow("lena denoised 32f", *d_lena_denoised_32fC1);
    }

    cv::waitKey();
  }
  catch (std::exception& e)
  {
    std::cout << "[exception] " << e.what() << std::endl;
    assert(false);
  }

  return EXIT_SUCCESS;

}

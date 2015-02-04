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

#include "default_msg.h"


int main(int /*argc*/, char** /*argv*/)
{
  try
  {
    imp::ImageCv8uC1 h1_lena_8uC1(cv::imread("/home/mwerlberger/data/std/Lena.tiff",
                                             CV_LOAD_IMAGE_GRAYSCALE),
                                  imp::PixelOrder::gray);

    // copy host->device
    std::shared_ptr<imp::cu::ImageGpu8uC1> d1_lena_8uC1(
          new imp::cu::ImageGpu8uC1(h1_lena_8uC1));

    // ROF denoising
    imp::cu::RofDenoising8uC1 rof;
    std::shared_ptr<imp::cu::ImageGpu8uC1> d_lena_denoised_8uC1(
          new imp::cu::ImageGpu8uC1(*d1_lena_8uC1));
    rof.denoise(d1_lena_8uC1, d_lena_denoised_8uC1);

    // copy denoised result back to host
    imp::ImageCv8uC1 h_lena_denoised_8uC1(*d_lena_denoised_8uC1);

    // show pictures
    cv::imshow("lena input", h1_lena_8uC1.cvMat());
    cv::imshow("lena denoised", h_lena_denoised_8uC1.cvMat());

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

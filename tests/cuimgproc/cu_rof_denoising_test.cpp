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

    rof.denoise(d_lena_denoised_8uC1, d1_lena_8uC1);
    std::cout << "\n"
              << rof
              << std::endl << std::endl;

    // copy denoised result back to host
    imp::ImageCv8uC1 h_lena_denoised_8uC1(*d_lena_denoised_8uC1);

    // show pictures
    cv::imshow("lena input", h1_lena_8uC1.cvMat());
    cv::imshow("lena denoised", h_lena_denoised_8uC1.cvMat());
    cv::waitKey();



    // 32fC1 test
    imp::ImageCv32fC1 h1_lena_32fC1(h1_lena_8uC1.size());

    h1_lena_8uC1.cvMat().convertTo(h1_lena_32fC1.cvMat(), CV_32F);
    h1_lena_32fC1.cvMat() /= 255.f;
    double min_val=0.0f, max_val=0.0f;
    cv::minMaxLoc(h1_lena_32fC1.cvMat(),  &min_val, &max_val);
    std::cout << "min: " << min_val << "; max: " << max_val << std::endl;

    std::shared_ptr<imp::cu::ImageGpu32fC1> d1_lena_32fC1(
          new imp::cu::ImageGpu32fC1(h1_lena_32fC1));
    std::shared_ptr<imp::cu::ImageGpu32fC1> d_lena_denoised_32fC1(
          new imp::cu::ImageGpu32fC1(*d1_lena_32fC1));

    imp::cu::RofDenoising32fC1 rof_32fC1;
    rof_32fC1.denoise(d_lena_denoised_32fC1, d1_lena_32fC1);
    imp::ImageCv32fC1 h_lena_denoised_32fC1(*d_lena_denoised_32fC1);

    cv::minMaxLoc(h_lena_denoised_32fC1.cvMat(),  &min_val, &max_val);
    std::cout << "denoised: min: " << min_val << "; max: " << max_val << std::endl;
//    cv::normalize(h_lena_denoised_32fC1.cvMat(),  h_lena_denoised_32fC1.cvMat(),
//                  0, 1, CV_MINMAX);
//    cv::minMaxLoc(h_lena_denoised_32fC1.cvMat(),  &min_val, &max_val);
//    std::cout << "denoised: min: " << min_val << "; max: " << max_val << std::endl;


    cv::imshow("lena input 32f", h1_lena_32fC1.cvMat());
    cv::imshow("lena denoised 32f", h_lena_denoised_32fC1.cvMat());
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

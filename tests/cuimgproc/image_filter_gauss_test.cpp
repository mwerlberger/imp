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
#include <imp/cuimgproc/cu_image_filter.cuh>

#include "default_msg.h"

int main(int /*argc*/, char** /*argv*/)
{
  try
  {
    imp::ImageCv8uC1 h1_lena_8uC1(cv::imread("/home/mwerlberger/data/std/Lena.tiff",
                                             CV_LOAD_IMAGE_GRAYSCALE),
                                  imp::PixelOrder::gray);

    {
      // copy host->device
      std::unique_ptr<imp::cu::ImageGpu8uC1> d_lena_8uC1(
            new imp::cu::ImageGpu8uC1(h1_lena_8uC1));

      std::unique_ptr<imp::cu::ImageGpu8uC1> d_gauss_lena_8uC1(
            new imp::cu::ImageGpu8uC1(d_lena_8uC1->size()));

      imp::cu::filterGauss(d_gauss_lena_8uC1.get(), d_lena_8uC1.get(), 10.0);


      imp::ImageCv8uC1 h_gauss_lena_8uC1(*d_gauss_lena_8uC1);
      cv::imshow("lena 8u", h1_lena_8uC1.cvMat());
      cv::imshow("lena gauss 8u", h_gauss_lena_8uC1.cvMat());

    }

    {
      // 32fC1 test
      imp::ImageCv32fC1 h1_lena_32fC1(h1_lena_8uC1.size());

      h1_lena_8uC1.cvMat().convertTo(h1_lena_32fC1.cvMat(), CV_32F);
      h1_lena_32fC1.cvMat() /= 255.f;

      // copy host->device
      std::unique_ptr<imp::cu::ImageGpu32fC1> d_lena_32fC1(
            new imp::cu::ImageGpu32fC1(h1_lena_32fC1));

      std::unique_ptr<imp::cu::ImageGpu32fC1> d_gauss_lena_32fC1(
            new imp::cu::ImageGpu32fC1(d_lena_32fC1->size()));

      imp::cu::filterGauss(d_gauss_lena_32fC1.get(), d_lena_32fC1.get(), 10.0);


      imp::ImageCv32fC1 h_gauss_lena_32fC1(*d_gauss_lena_32fC1);
      cv::imshow("lena 32f", h1_lena_32fC1.cvMat());
      cv::imshow("lena gauss 32f", h_gauss_lena_32fC1.cvMat());

    }
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

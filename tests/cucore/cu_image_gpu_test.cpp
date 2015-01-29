#include <assert.h>
#include <cstdint>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <imp/core/roi.hpp>
#include <imp/core/image_raw.hpp>
#include <imp/core/image_cv.hpp>
#include <imp/cucore/cu_image_gpu.cuh>

#include "default_msg.h"


int main(int /*argc*/, char** /*argv*/)
{
  try
  {
    cv::Mat lena = cv::imread("/home/mwerlberger/data/std/Lena.pgm", CV_LOAD_IMAGE_GRAYSCALE);

    imp::Size2u sz(513, 512);
    imp::cu::ImageGpu8uC1 im8uC1(sz);

    std::cout << "im8uC1: " << im8uC1 << std::endl;

    imp::ImageRaw8uC1 h1_lena_8uC1(
          reinterpret_cast<imp::ImageRaw8uC1::pixel_container_t>(lena.data),
          lena.cols, lena.rows, lena.step, true);

    // copy host->device->device->host
    imp::cu::ImageGpu8uC1 d1_lena_8uC1(h1_lena_8uC1);
    imp::cu::ImageGpu8uC1 d2_lena_8uC1(d1_lena_8uC1);
    imp::ImageCv8uC1 h2_lena_8uC1(d2_lena_8uC1);

    std::cout << "h1_lena_8uC1: " << h1_lena_8uC1 << std::endl;
    std::cout << "h2_lena_8uC1: " << h2_lena_8uC1 << std::endl;
    std::cout << "d1_lena_8uC1: " << d1_lena_8uC1 << std::endl;
    std::cout << "d2_lena_8uC1: " << d2_lena_8uC1 << std::endl;

    cv::imshow("input", lena);
    cv::imshow("copy", h2_lena_8uC1.cvMat());
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

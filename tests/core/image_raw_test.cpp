#include <assert.h>
#include <cstdint>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <imp/core/roi.hpp>
#include <imp/core/image_raw.hpp>

#include "default_msg.h"


int main(int /*argc*/, char** /*argv*/)
{
  try
  {
    cv::Mat lena = cv::imread("/home/mwerlberger/data/std/Lena.pgm", CV_LOAD_IMAGE_GRAYSCALE);

    imp::Size2u sz(513, 512);
    imp::ImageRaw8uC1 im8uC1(sz);

    std::cout << "im8uC1: " << im8uC1 << std::endl;

    imp::ImageRaw8uC1 lena_8uC1(
          reinterpret_cast<imp::ImageRaw8uC1::pixel_container_t>(lena.data),
          lena.cols, lena.rows, lena.step, true);
    imp::ImageRaw8uC1 lena_copy_8uC1(lena_8uC1);
    //lena_8uC1.copyTo(lena_copy_8uC1);

    std::cout << "lena_8uC1:      " << lena_8uC1 << std::endl;
    std::cout << "lena_copy_8uC1: " << lena_copy_8uC1 << std::endl;


    cv::Mat lena_copy_mat(lena_copy_8uC1.height(), lena_copy_8uC1.width(),
                          CV_8UC1, lena_copy_8uC1.data(), lena_copy_8uC1.pitch());

    cv::imshow("input", lena);
    cv::imshow("copy", lena_copy_mat);
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

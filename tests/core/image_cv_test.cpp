#include <assert.h>
#include <cstdint>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <imp/core/roi.hpp>
#include <imp/core/image_cv.hpp>

#include "default_msg.h"


int main(int /*argc*/, char** /*argv*/)
{
  try
  {
    cv::Mat lena = cv::imread("/home/mwerlberger/data/std/Lena.pgm", CV_LOAD_IMAGE_GRAYSCALE);

    imp::Size2u sz(510, 512);
    imp::ImageCv8uC1 im8uC1(sz);

    std::cout << "im8uC1: " << im8uC1 << std::endl;

    imp::ImageCv8uC1 lena_8uC1(lena);
    imp::ImageCv8uC1 lena_copy_8uC1(lena_8uC1);
    //lena_8uC1.copyTo(lena_copy_8uC1);

    std::cout << "lena_8uC1:      " << lena_8uC1 << std::endl;
    std::cout << "lena_copy_8uC1: " << lena_copy_8uC1 << std::endl;


    cv::imshow("input", lena_8uC1.cvMat());
    cv::imshow("copy", lena_copy_8uC1.cvMat());
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

#include <assert.h>
#include <cstdint>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <imp/core/roi.hpp>
#include <imp/image/image_raw.hpp>
#include <imp/bridge/opencv/image_cv.hpp>
#include <imp/cucore/cu_image_gpu.cuh>

#include "default_msg.h"


int main(int /*argc*/, char** /*argv*/)
{
  try
  {
//    cv::Mat lena = cv::imread("/home/mwerlberger/data/std/Lena.pgm", CV_LOAD_IMAGE_GRAYSCALE);


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


    // set-value test:
    imp::Size2u sz(513, 512);
    imp::cu::ImageGpu8uC1 d0_8uC1(sz);
    d0_8uC1.setValue(imp::Pixel8uC1(128));
    imp::ImageCv8uC1 h0_8uC1(d0_8uC1);

    imp::cu::ImageGpu8uC3 d1_8uC3(sz);
    d1_8uC3.setValue(imp::Pixel8uC3(255, 0, 0));
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

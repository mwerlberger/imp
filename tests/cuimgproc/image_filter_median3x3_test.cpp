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


void addImpulseNoise(cv::Mat& img, double perc)
{
  const int rows = img.rows;
  const int cols = img.cols;
  const int channels = img.channels();
  int num_corrupted_pts = static_cast<int>((rows*cols*channels)*perc/100.0);

  for (int i=0; i<num_corrupted_pts; ++i)
  {
    int r = rand() % rows;
    int c = rand() % cols;
    int channel = rand() % channels;

    uchar* pixel = img.ptr<uchar>(r) + (c*channels) + channel;
    *pixel = (rand()%2) ? 255 : 0;
  }
}

int main(int /*argc*/, char** /*argv*/)
{
  try
  {
    imp::ImageCv8uC1 h1_lena_8uC1(cv::imread("/home/mwerlberger/data/std/Lena.tiff",
                                             CV_LOAD_IMAGE_GRAYSCALE),
                                  imp::PixelOrder::gray);

    // add salt and pepper noise
    addImpulseNoise(h1_lena_8uC1.cvMat(), 20);

    {
      // copy host->device
      std::unique_ptr<imp::cu::ImageGpu8uC1> d_lena_8uC1(
            new imp::cu::ImageGpu8uC1(h1_lena_8uC1));

      std::unique_ptr<imp::cu::ImageGpu8uC1> d_median_lena_8uC1(
            new imp::cu::ImageGpu8uC1(d_lena_8uC1->size()));

      imp::cu::filterMedian3x3(d_median_lena_8uC1.get(), d_lena_8uC1.get());


      imp::ImageCv8uC1 h_median_lena_8uC1(*d_median_lena_8uC1);
      cv::imshow("lena 8u", h1_lena_8uC1.cvMat());
      cv::imshow("lena median 8u", h_median_lena_8uC1.cvMat());

      cv::waitKey();
    }
  }
  catch (std::exception& e)
  {
    std::cout << "[exception] " << e.what() << std::endl;
    assert(false);
  }

  std::cout << imp::ok_msg << std::endl;

  return EXIT_SUCCESS;

}

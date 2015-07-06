#include <assert.h>
#include <cstdint>
#include <iostream>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <imp/core/roi.hpp>
#include <imp/core/image_raw.hpp>
#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/cu_imgproc/iterative_kernel_calls.cuh>

int main(int argc, char** argv)
{
  try
  {
    //std::cout << "usage: texture_issue [break_things_flag]";

    // dummy test data
    imp::Size2u sz(250,250);
    imp::Roi2u roi(sz.width()/3, sz.height()/3, sz.width()/3, sz.height()/3);
    imp::ImageRaw32fC1 image(sz);
    imp::ImageRaw32fC1 result_image(sz);
    image.setValue(0.0f);
    result_image.setValue(0.0f);

    for(size_t y=roi.y(); y<roi.y()+roi.height(); ++y)
    {
      for(size_t x=roi.x(); x<roi.x()+roi.width(); ++x)
      {
        image[y][x] = 1.0f;
      }
    }

    imp::cu::ImageGpu32fC1::Ptr cu_image = std::make_shared<imp::cu::ImageGpu32fC1>(image.width(), image.height());//(image.cols, image.rows);
    imp::cu::ImageGpu32fC1::Ptr cu_result_image = std::make_shared<imp::cu::ImageGpu32fC1>(image.width(), image.height());//(image.cols, image.rows);
    image.copyTo(*cu_image);
    IMP_CUDA_CHECK();

    //
    //
    imp::cu::IterativeKernelCalls ikc;
    bool break_things = (argc>1) ? true : false;
    ikc.run(cu_result_image, cu_image, break_things);
    IMP_CUDA_CHECK();
    //
    //

    cu_result_image->copyTo(result_image);
    IMP_CUDA_CHECK();

    float in_sum = 0.f;
    float out_sum = 0.f;
    for(size_t y=0; y<result_image.height(); ++y)
    {
      for(size_t x=0; x<result_image.width(); ++x)
      {
        in_sum += image[y][x];
        out_sum += result_image[y][x];
      }
    }
    std::cout << "in_sum: " << in_sum << "; out_sum: " << out_sum << std::endl;

    cv::Mat vis_in(image.height(), image.width(), CV_32FC1, image.data(), image.pitch());
    cv::Mat vis_out(result_image.height(), result_image.width(), CV_32FC1, result_image.data(), result_image.pitch());

    cv::imshow("input", vis_in);
    cv::imshow("output", vis_out);
    cv::waitKey();

  }
  catch (std::exception& e)
  {
    std::cout << "[exception] " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;

}

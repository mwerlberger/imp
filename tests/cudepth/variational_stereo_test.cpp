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
#include <imp/cucore/cu_math.cuh>

#include <imp/cudepth/variational_stereo.hpp>

#include "default_msg.h"


int main(int /*argc*/, char** /*argv*/)
{
  try
  {
    imp::ImageCv8uC1 h_cones1_8uC1(
          cv::imread("/home/mwerlberger/data/std/cones/im2.ppm",
                     CV_LOAD_IMAGE_GRAYSCALE), imp::PixelOrder::gray);
    imp::ImageCv8uC1 h_cones2_8uC1(
          cv::imread("/home/mwerlberger/data/std/cones/im6.ppm",
                     CV_LOAD_IMAGE_GRAYSCALE), imp::PixelOrder::gray);

    // 8u -> 32f

    imp::ImageCv32fC1 h_cones1_32fC1(h_cones1_8uC1.size());
    imp::ImageCv32fC1 h_cones2_32fC1(h_cones2_8uC1.size());

    h_cones1_8uC1.cvMat().convertTo(h_cones1_32fC1.cvMat(), CV_32F);
    h_cones1_32fC1.cvMat() /= 255.f;
    h_cones2_8uC1.cvMat().convertTo(h_cones2_32fC1.cvMat(), CV_32F);
    h_cones2_32fC1.cvMat() /= 255.f;

    // host -> device

    std::shared_ptr<imp::cu::ImageGpu32fC1> d_cones1_32fC1(
          new imp::cu::ImageGpu32fC1(h_cones1_32fC1));
    std::shared_ptr<imp::cu::ImageGpu32fC1> d_cones2_32fC1(
          new imp::cu::ImageGpu32fC1(h_cones2_32fC1));

    std::unique_ptr<imp::cu::VariationalStereo> stereo(
          new imp::cu::VariationalStereo());

    stereo->addImage(d_cones1_32fC1);
    stereo->addImage(d_cones2_32fC1);

    stereo->solve();

    std::shared_ptr<imp::cu::ImageGpu32fC1> disp = stereo->getDisparities();
    imp::Pixel32fC1 min_val,max_val;
    imp::cu::minMax(disp, min_val, max_val);

    std::cout << "disp: min: " << min_val.x << " max: " << max_val.x << std::endl;

    imp::ImageCv32fC1 h_disp(*disp);
    h_disp.cvMat() = (h_disp.cvMat() - min_val.x)/(max_val.x-min_val.x);


    cv::imshow("cones1", h_cones1_32fC1.cvMat());
    cv::imshow("cones2", h_cones2_32fC1.cvMat());
    cv::imshow("disp", h_disp.cvMat());

    //cv::imshow("cones1 denoised 32f", h_cones1_denoised_32fC1.cvMat());

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

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
#include <imp/cuimgproc/edge_detectors.cuh>
#include <imp/bridge/opencv/cu_cv_bridge.hpp>


int main(int /*argc*/, char** /*argv*/)
{
  try
  {
    std::shared_ptr<imp::cu::ImageGpu32fC1> im;
    imp::cu::cvBridgeLoad(im, "/home/mwerlberger/data/std/cones/im2.ppm",
                          imp::PixelOrder::gray);

    std::unique_ptr<imp::cu::ImageGpu32fC1> edges(
          new imp::cu::ImageGpu32fC1(*im));

    imp::cu::naturalEdges(*edges, *im, 1.f, 10.f, 0.7f);

    imp::cu::cvBridgeShow("image", *im);
    imp::cu::cvBridgeShow("edges", *edges, true);

    cv::waitKey();
  }
  catch (std::exception& e)
  {
    std::cout << "[exception] " << e.what() << std::endl;
    assert(false);
  }

  return EXIT_SUCCESS;

}

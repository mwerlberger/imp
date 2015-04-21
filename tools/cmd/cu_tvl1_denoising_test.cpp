#include <assert.h>
#include <cstdint>
#include <iostream>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <imp/core/roi.hpp>
#include <imp/image/image_raw.hpp>
#include <imp/bridge/opencv/image_cv.hpp>
#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/cu_imgproc/cu_tvl1_denoising.cuh>
#include <imp/bridge/opencv/cu_cv_bridge.hpp>

int main(int /*argc*/, char** /*argv*/)
{
  try
  {
    // 8uC1
    {
      std::shared_ptr<imp::cu::ImageGpu8uC1> d_lena;
      imp::cu::cvBridgeLoad(d_lena, "/home/mwerlberger/data/std/Lena.tiff",
                            imp::PixelOrder::gray);
      std::shared_ptr<imp::cu::ImageGpu8uC1> d_lena_denoised(
            new imp::cu::ImageGpu8uC1(*d_lena));

      imp::cu::TvL1Denoising8uC1 tvl1;
      tvl1.params().lambda = 0.5f;
      std::cout << "\n" << tvl1 << std::endl << std::endl;
      tvl1.denoise(d_lena_denoised, d_lena);

      // show results
      imp::cu::cvBridgeShow("lena input 8u", *d_lena);
      imp::cu::cvBridgeShow("lena denoised 8u", *d_lena_denoised);
    }

    // 32fC1
    {
      std::shared_ptr<imp::cu::ImageGpu32fC1> d1_lena;
      imp::cu::cvBridgeLoad(d1_lena, "/home/mwerlberger/data/std/Lena.tiff",
                            imp::PixelOrder::gray);
      std::shared_ptr<imp::cu::ImageGpu32fC1> d_lena_denoised(
            new imp::cu::ImageGpu32fC1(*d1_lena));

      imp::cu::TvL1Denoising32fC1 tvl1;
      tvl1.params().lambda = 0.5f;
      std::cout << "\n" << tvl1 << std::endl << std::endl;
      tvl1.denoise(d_lena_denoised, d1_lena);

      imp::cu::cvBridgeShow("lena input 32f", *d1_lena);
      imp::cu::cvBridgeShow("lena denoised 32f", *d_lena_denoised);
    }

    cv::waitKey();
  }
  catch (std::exception& e)
  {
    std::cout << "[exception] " << e.what() << std::endl;
    assert(false);
  }

  return EXIT_SUCCESS;

}

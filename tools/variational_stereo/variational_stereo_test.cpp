#include <assert.h>
#include <cstdint>
#include <iostream>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <imp/core/roi.hpp>
#include <imp/core/image_raw.hpp>
#include <imp/bridge/opencv/image_cv.hpp>
#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/cu_core/cu_math.cuh>
#include <imp/bridge/opencv/cu_cv_bridge.hpp>

#include <imp/cu_correspondence/variational_stereo.hpp>

int main(int /*argc*/, char** /*argv*/)
{
  using Stereo = imp::cu::VariationalStereo;
  using StereoParameters = Stereo::Parameters;

  try
  {
    imp::cu::ImageGpu32fC1::Ptr d_cones1_32fC1;
    imp::cu::cvBridgeLoad(d_cones1_32fC1, "/home/mwerlberger/data/std/cones/im2.ppm",
                          imp::PixelOrder::gray);
    imp::cu::ImageGpu32fC1::Ptr d_cones2_32fC1;
    imp::cu::cvBridgeLoad(d_cones2_32fC1, "/home/mwerlberger/data/std/cones/im6.ppm",
                          imp::PixelOrder::gray);

    {
      imp::Pixel32fC1 min_val,max_val;
      imp::cu::minMax(*d_cones1_32fC1, min_val, max_val);
      std::cout << "disp: min: " << min_val.x << " max: " << max_val.x << std::endl;
    }


    StereoParameters::Ptr stereo_params = std::make_shared<StereoParameters>();
    stereo_params->verbose = 0;
    stereo_params->solver = imp::cu::StereoPDSolver::PrecondHuberL1Weighted;
    stereo_params->ctf.scale_factor = 0.8f;
    stereo_params->ctf.iters = 50;
    stereo_params->ctf.warps  = 10;
    stereo_params->ctf.apply_median_filter = true;

    std::unique_ptr<Stereo> stereo(new Stereo(stereo_params));

    stereo->addImage(d_cones1_32fC1);
    stereo->addImage(d_cones2_32fC1);

    stereo->solve();

    imp::cu::ImageGpu32fC1::Ptr d_disp = stereo->getDisparities();
    imp::cu::ImageGpu32fC1::Ptr d_occ = stereo->getOcclusion();

    {
      imp::Pixel32fC1 min_val,max_val;
      imp::cu::minMax(*d_disp, min_val, max_val);
      std::cout << "disp: min: " << min_val.x << " max: " << max_val.x << std::endl;
    }

    imp::cu::cvBridgeShow("cones im2", *d_cones1_32fC1);
    imp::cu::cvBridgeShow("cones im6", *d_cones2_32fC1);
    *d_disp *= -1;
    {
      imp::Pixel32fC1 min_val,max_val;
      imp::cu::minMax(*d_disp, min_val, max_val);
      std::cout << "disp: min: " << min_val.x << " max: " << max_val.x << std::endl;
    }

    imp::cu::cvBridgeShow("disparities", *d_disp, true);

    if (d_occ)
    {
      imp::cu::cvBridgeShow("occlusions", *d_occ, true);
    }

    // get primal energy
    imp::cu::ImageGpu32fC1::Ptr d_ep = stereo->computePrimalEnergy();
    imp::cu::cvBridgeShow("primal energy", *d_ep, true);
    imp::Pixel32fC1 ep_min, ep_max;
    imp::cu::minMax(*d_ep, ep_min, ep_max);
    std::cout << "primal energy: min: " << ep_min.x << " max: " << ep_max.x << std::endl;


    cv::waitKey();
  }
  catch (std::exception& e)
  {
    std::cout << "[exception] " << e.what() << std::endl;
    assert(false);
  }

  return EXIT_SUCCESS;

}

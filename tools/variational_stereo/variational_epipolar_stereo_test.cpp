#include <assert.h>
#include <cstdint>
#include <iostream>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <imp/core/roi.hpp>
#include <imp/core/image_raw.hpp>
#include <imp/bridge/opencv/image_cv.hpp>
#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/cu_core/cu_math.cuh>
#include <imp/bridge/opencv/cu_cv_bridge.hpp>

#include <imp/cu_core/cu_se3.cuh>
#include <imp/cu_core/cu_matrix.cuh>
#include <imp/cu_core/cu_pinhole_camera.cuh>

#include <imp/cu_correspondence/variational_epipolar_stereo.hpp>

imp::ImageCv32fC1::Ptr loadUint4ToFloat(const std::string& filename)
{
  cv::Mat im_as_4uint = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat im_32f(im_as_4uint.rows, im_as_4uint.cols/4, CV_32F, im_as_4uint.data);
  imp::ImageCv32fC1::Ptr img(new imp::ImageCv32fC1(im_32f.clone()));
  return img;
  //img.reset(new imp::ImageCv32fC1(im_32f.clone()));
}

int main(int /*argc*/, char** /*argv*/)
{
  try
  {
    imp::ImageCv32fC1::Ptr cv_im0;
    imp::cvBridgeLoad(cv_im0,
                      //"/home/mwerlberger/data/epipolar_stereo_test/00000.png",
                      "/home/mwerlberger/data/remode_1436355983/0_image_.png",
                      imp::PixelOrder::gray);

    imp::ImageCv32fC1::Ptr cv_im1;
    imp::cvBridgeLoad(cv_im1,
                      //"/home/mwerlberger/data/epipolar_stereo_test/00001.png",
                      "/home/mwerlberger/data/remode_1436355983/1_image_.png",
                      imp::PixelOrder::gray);

    // rectify images for testing

    imp::cu::ImageGpu32fC1::Ptr im0 = std::make_shared<imp::cu::ImageGpu32fC1>(*cv_im0);
    imp::cu::ImageGpu32fC1::Ptr im1 = std::make_shared<imp::cu::ImageGpu32fC1>(*cv_im1);

    imp::ImageCv32fC1::Ptr cv_a = loadUint4ToFloat("/home/mwerlberger/data/remode_1436355983/47_a_.png");
    imp::cu::ImageGpu32fC1::Ptr in_a = std::make_shared<imp::cu::ImageGpu32fC1>(*cv_a);
    imp::ImageCv32fC1::Ptr cv_b = loadUint4ToFloat("/home/mwerlberger/data/remode_1436355983/47_b_.png");
    imp::cu::ImageGpu32fC1::Ptr in_b = std::make_shared<imp::cu::ImageGpu32fC1>(*cv_b);
    imp::ImageCv32fC1::Ptr cv_mu = loadUint4ToFloat("/home/mwerlberger/data/remode_1436355983/47_mu_.png");
    imp::cu::ImageGpu32fC1::Ptr in_mu = std::make_shared<imp::cu::ImageGpu32fC1>(*cv_mu);

    imp::ImageCv32fC1::Ptr cv_sigma2 = loadUint4ToFloat("/home/mwerlberger/data/remode_1436355983/47_sigma2_.png");
    //imp::cu::ImageGpu32fC1::Ptr in_sigma2 = std::make_shared<imp::cu::ImageGpu32fC1>(*cv_sigma2);

    double min_sigma2, max_sigma2;
    cv::minMaxLoc(cv_sigma2->cvMat(), &min_sigma2, &max_sigma2);
    double min_a, max_a;
    cv::minMaxLoc(cv_a->cvMat(), &min_a, &max_a);
    double min_b, max_b;
    cv::minMaxLoc(cv_b->cvMat(), &min_b, &max_b);

    std::cout << "sigma2: [" << min_sigma2 << ", " << max_sigma2 << "]; "
              << "a: [" << min_a << ", " << max_a << "]; "
              << "b: [" << min_b << ", " << max_b << "]; "
              << std::endl;
    imp::ImageCv32fC1::Ptr confidence = std::make_shared<imp::ImageCv32fC1>(cv_a->size());
    for (std::uint32_t y=0; y<confidence->height(); ++y)
    {
      for (std::uint32_t x=0; x<confidence->width(); ++x)
      {
//        float sigma2 = (*cv_sigma2)[y][x];
        float a = (*cv_a)[y][x];
        float b = (*cv_b)[y][x];
//        (*confidence)[y][x] = .5f * (max_sigma2-sigma2)/max_sigma2;
//        (*confidence)[y][x] = 10.f* (a-min_a)/(max_a-min_a);
        float ab = a/b;
        (*confidence)[y][x] = std::max(.0f, std::exp(ab/1.4f)-1.f);
      }
    }

    double min_confidence, max_confidence;
    cv::minMaxLoc(confidence->cvMat(), &min_confidence, &max_confidence);
    std::cout << "confidence: [" << min_confidence << ", " << max_confidence << "]; " << std::endl;


//    Eigen::Quaterniond q_world_im0(0.14062777, 0.98558398, 0.02351040, -0.09107859);
//    Eigen::Quaterniond q_world_im1(0.14118687, 0.98569744, 0.01930722, -0.08996696);
//    Eigen::Vector3d t_world_im0(-0.12617580, 0.50447008, 0.15342121);
//    Eigen::Vector3d t_world_im1(-0.11031053, 0.50314023, 0.15158643);
    Eigen::Quaterniond q_world_im0(-0.189093259866752489, 0.973026745106075341,
                                   0.122442858311335945, -0.0497035092301516962);
    Eigen::Quaterniond q_world_im1(-0.18906950845598447, 0.97263493728776107,
                                   0.125280086688952852, -0.0503874946653839265);
    Eigen::Vector3d t_world_im0(0.183387220147751662, -0.216931228877717486, -0.389589144279252619);
    Eigen::Vector3d t_world_im1(0.17447576240725754, -0.218695106328406164, -0.389681016821063875);

    imp::cu::PinholeCamera cu_cam(414.09, 413.799, 355.567, 246.337);

    // im0: fixed image; im1: moving image
    imp::cu::Matrix3f F_fix_mov;
    imp::cu::Matrix3f F_mov_fix;
    Eigen::Matrix3d F_fm, F_mf;
    { // compute fundamental matrix
      Eigen::Matrix3d R_world_mov = q_world_im1.matrix();
      Eigen::Matrix3d R_world_fix = q_world_im0.matrix();
      Eigen::Matrix3d R_fix_mov = R_world_fix.inverse()*R_world_mov;

      // in ref coordinates
      Eigen::Vector3d t_fix_mov = R_world_fix.inverse()*(-t_world_im0 + t_world_im1);

      Eigen::Matrix3d tx_fix_mov;
      tx_fix_mov << 0, -t_fix_mov[2], t_fix_mov[1],
          t_fix_mov[2], 0, -t_fix_mov[0],
          -t_fix_mov[1], t_fix_mov[0], 0;
      Eigen::Matrix3d E_fix_mov = tx_fix_mov * R_fix_mov;
      Eigen::Matrix3d K;
      K << cu_cam.fx(), 0, cu_cam.cx(),
          0, cu_cam.fy(), cu_cam.cy(),
          0, 0, 1;

      Eigen::Matrix3d Kinv = K.inverse();
      F_fm = Kinv.transpose() * E_fix_mov * Kinv;
      F_mf = F_fm.transpose();
    } // end .. compute fundamental matrix
    // convert the Eigen-thingy to something that we can use in CUDA
    for(size_t row=0; row<F_fix_mov.rows(); ++row)
    {
      for(size_t col=0; col<F_fix_mov.cols(); ++col)
      {
        F_fix_mov(row,col) = (float)F_fm(row,col);
        F_mov_fix(row,col) = (float)F_mf(row,col);
      }
    }

    // compute SE3 transformation
    imp::cu::SE3<float> T_world_fix(
          static_cast<float>(q_world_im0.w()), static_cast<float>(q_world_im0.x()),
          static_cast<float>(q_world_im0.y()), static_cast<float>(q_world_im0.z()),
          static_cast<float>(t_world_im0.x()), static_cast<float>(t_world_im0.y()),
          static_cast<float>(t_world_im0.z()));
    imp::cu::SE3<float> T_world_mov(
          static_cast<float>(q_world_im1.w()), static_cast<float>(q_world_im1.x()),
          static_cast<float>(q_world_im1.y()), static_cast<float>(q_world_im1.z()),
          static_cast<float>(t_world_im1.x()), static_cast<float>(t_world_im1.y()),
          static_cast<float>(t_world_im1.z()));
    imp::cu::SE3<float> T_mov_fix = T_world_mov.inv() * T_world_fix;
    // end .. compute SE3 transformation

    std::cout << "T_mov_fix:\n" << T_mov_fix << std::endl;


    std::unique_ptr<imp::cu::VariationalEpipolarStereo> stereo(
          new imp::cu::VariationalEpipolarStereo());

    stereo->parameters()->verbose = 1;
    stereo->parameters()->solver = imp::cu::StereoPDSolver::EpipolarPrecondHuberL1;
    stereo->parameters()->ctf.scale_factor = 0.8f;
    stereo->parameters()->ctf.iters = 20;
    stereo->parameters()->ctf.warps  = 10;
    stereo->parameters()->ctf.apply_median_filter = true;
    //stereo->parameters()->lambda = 15;
    // pointwise lambda with according to confidences
    imp::cu::ImageGpu32fC1::Ptr lambda = std::make_shared<imp::cu::ImageGpu32fC1>(*confidence);
    stereo->parameters()->lambda_pointwise = lambda;


    stereo->addImage(im0);
    stereo->addImage(im1);

    imp::cu::ImageGpu32fC1::Ptr cu_mu = std::make_shared<imp::cu::ImageGpu32fC1>(im0->size());
    imp::cu::ImageGpu32fC1::Ptr cu_sigma2 = std::make_shared<imp::cu::ImageGpu32fC1>(im0->size());
    cu_mu->setValue(1.f);
    cu_sigma2->setValue(0.0f);

    stereo->setFundamentalMatrix(F_mov_fix);
    stereo->setIntrinsics({cu_cam, cu_cam});
    stereo->setExtrinsics(T_mov_fix);
    stereo->setDepthProposal(cu_mu, cu_sigma2);

    stereo->solve();

    std::shared_ptr<imp::cu::ImageGpu32fC1> d_disp = stereo->getDisparities();


    {
      imp::Pixel32fC1 min_val,max_val;
      imp::cu::minMax(*d_disp, min_val, max_val);
      std::cout << "disp: min: " << min_val.x << " max: " << max_val.x << std::endl;
    }

    imp::cu::cvBridgeShow("im0", *im0);
    imp::cu::cvBridgeShow("im1", *im1);
//    *d_disp *= -1;
    {
      imp::Pixel32fC1 min_val,max_val;
      imp::cu::minMax(*d_disp, min_val, max_val);
      std::cout << "disp: min: " << min_val.x << " max: " << max_val.x << std::endl;
    }

    imp::cu::cvBridgeShow("disparities", *d_disp, -3.0f, 6.0f);
    imp::cu::cvBridgeShow("disparities minmax", *d_disp, true);

    imp::cu::cvBridgeShow("a (converged)", *in_a, true);
    imp::cu::cvBridgeShow("b (converged)", *in_b, true);
    imp::cu::cvBridgeShow("mu (converged)", *in_mu, true);
    imp::cvBridgeShow("sigma2 (converged)", *cv_sigma2, true);
    imp::cu::cvBridgeShow("lambda (converged)", *lambda, true);


    cv::waitKey();
  }
  catch (std::exception& e)
  {
    std::cout << "[exception] " << e.what() << std::endl;
    assert(false);
  }

  return EXIT_SUCCESS;

}

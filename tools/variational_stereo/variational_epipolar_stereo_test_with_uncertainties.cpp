#include <assert.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <fstream>

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

int main(int argc, char** argv)
{
  if (argc<2)
  {
    std::cout << "usage: " << argv[0] << " input_directory [output_directory]" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string input_directory(argv[1]);
  std::string output_directory(input_directory + "/out");
  if (argc>2)
    output_directory = std::string(argv[2]);


  std::string images_file(input_directory + "/images.txt");
  std::ifstream dataset_fs(images_file.c_str());
  if(!dataset_fs.is_open())
  {
    std::cout << "Could not open images file: " << images_file << std::endl;
    return EXIT_FAILURE;
  }

  imp::ImageCv32fC1::Ptr cv_image_ref;
  imp::ImageCv32fC1::Ptr cv_image_mov;

  while(dataset_fs.good() && !dataset_fs.eof())
  {
    // skip comments
    if(dataset_fs.peek() == '#')
      dataset_fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // load data
    size_t img_id;
    double ts, tx, ty, tz, qx, qy, qz, qw;
    dataset_fs >> ts >> img_id >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    Eigen::Vector3d t_world_cur(tx, ty, tz);
    Eigen::Quaterniond q_world_cur(qw, qx, qy, qz);
    q_world_cur.normalize();
    std::stringstream img_name;
    img_name << input_directory << "/" << img_id << "_image_.png";

    std::cout << "reading " << img_name.str() << std::endl;

    if (!cv_image_ref || img_id == 0)
    {
      imp::cvBridgeLoad(cv_image_ref, img_name.str(), imp::PixelOrder::gray);
      imp::cvBridgeShow("image_ref", *cv_image_ref);
      cv::waitKey();
      continue;
    }
    imp::cvBridgeLoad(cv_image_mov, img_name.str(), imp::PixelOrder::gray);
    img_name.str("");

    img_name << input_directory << "/" << img_id << "_a_.png";
    std::cout << "reading " << img_name.str() << std::endl;
    cv::Mat a_as_4uint = cv::imread(img_name.str(), CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat a_32f(a_as_4uint.rows, a_as_4uint.cols/4, CV_32F, a_as_4uint.data);
//    cv::imshow("opencv a (4uint)", a_as_4uint);
//    cv::imshow("opencv a", a_32f);
    imp::ImageCv32fC1 a(a_32f);
    imp::cvBridgeShow("a", a);
    imp::cu::ImageGpu32fC1 cu_a(a);
    imp::cu::cvBridgeShow("cu_a", cu_a);
    img_name.str("");

    img_name << input_directory << "/" << img_id << "_mu_.png";
    std::cout << "reading " << img_name.str() << std::endl;
    cv::Mat mu_as_4uint = cv::imread(img_name.str(), CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat mu_32f(mu_as_4uint.rows, mu_as_4uint.cols/4, CV_32F, mu_as_4uint.data);
    imp::ImageCv32fC1 mu(mu_32f);
    imp::cvBridgeShow("mu", mu, true);
    imp::cu::ImageGpu32fC1 cu_mu(mu);
    imp::cu::cvBridgeShow("cu_mu", cu_mu, true);
    img_name.str("");


    cv::waitKey();
  }



//    imp::cu::ImageGpu32fC1::Ptr im0 = std::make_shared<imp::cu::ImageGpu32fC1>(*cv_im0);
//    imp::cu::ImageGpu32fC1::Ptr im1 = std::make_shared<imp::cu::ImageGpu32fC1>(*cv_im1);


//    Eigen::Quaterniond q_world_im0(0.14062777, 0.98558398, 0.02351040, -0.09107859);
//    Eigen::Quaterniond q_world_im1(0.14118687, 0.98569744, 0.01930722, -0.08996696);
//    Eigen::Vector3d t_world_im0(-0.12617580, 0.50447008, 0.15342121);
//    Eigen::Vector3d t_world_im1(-0.11031053, 0.50314023, 0.15158643);

//    imp::cu::PinholeCamera cu_cam(414.09, 413.799, 355.567, 246.337);

//    // im0: fixed image; im1: moving image
//    imp::cu::Matrix3f F_fix_mov;
//    imp::cu::Matrix3f F_mov_fix;
//    Eigen::Matrix3d F_fm, F_mf;
//    { // compute fundamental matrix
//      Eigen::Matrix3d R_world_mov = q_world_im1.matrix();
//      Eigen::Matrix3d R_world_fix = q_world_im0.matrix();
//      Eigen::Matrix3d R_fix_mov = R_world_fix.inverse()*R_world_mov;

//      // in ref coordinates
//      Eigen::Vector3d t_fix_mov = R_world_fix.inverse()*(-t_world_im0 + t_world_im1);

//      Eigen::Matrix3d tx_fix_mov;
//      tx_fix_mov << 0, -t_fix_mov[2], t_fix_mov[1],
//          t_fix_mov[2], 0, -t_fix_mov[0],
//          -t_fix_mov[1], t_fix_mov[0], 0;
//      Eigen::Matrix3d E_fix_mov = tx_fix_mov * R_fix_mov;
//      Eigen::Matrix3d K;
//      K << cu_cam.fx(), 0, cu_cam.cx(),
//          0, cu_cam.fy(), cu_cam.cy(),
//          0, 0, 1;

//      Eigen::Matrix3d Kinv = K.inverse();
//      F_fm = Kinv.transpose() * E_fix_mov * Kinv;
//      F_mf = F_fm.transpose();
//    } // end .. compute fundamental matrix
//    // convert the Eigen-thingy to something that we can use in CUDA
//    for(size_t row=0; row<F_fix_mov.rows(); ++row)
//    {
//      for(size_t col=0; col<F_fix_mov.cols(); ++col)
//      {
//        F_fix_mov(row,col) = (float)F_fm(row,col);
//        F_mov_fix(row,col) = (float)F_mf(row,col);
//      }
//    }

//    // compute SE3 transformation
//    imp::cu::SE3<float> T_world_fix(
//          static_cast<float>(q_world_im0.w()), static_cast<float>(q_world_im0.x()),
//          static_cast<float>(q_world_im0.y()), static_cast<float>(q_world_im0.z()),
//          static_cast<float>(t_world_im0.x()), static_cast<float>(t_world_im0.y()),
//          static_cast<float>(t_world_im0.z()));
//    imp::cu::SE3<float> T_world_mov(
//          static_cast<float>(q_world_im1.w()), static_cast<float>(q_world_im1.x()),
//          static_cast<float>(q_world_im1.y()), static_cast<float>(q_world_im1.z()),
//          static_cast<float>(t_world_im1.x()), static_cast<float>(t_world_im1.y()),
//          static_cast<float>(t_world_im1.z()));
//    imp::cu::SE3<float> T_mov_fix = T_world_mov.inv() * T_world_fix;
//    // end .. compute SE3 transformation

//    std::cout << "T_mov_fix:\n" << T_mov_fix << std::endl;


//    std::unique_ptr<imp::cu::VariationalEpipolarStereo> stereo(
//          new imp::cu::VariationalEpipolarStereo());

//    stereo->parameters()->verbose = 1;
//    stereo->parameters()->solver = imp::cu::StereoPDSolver::EpipolarPrecondHuberL1;
//    stereo->parameters()->ctf.scale_factor = 0.8f;
//    stereo->parameters()->ctf.iters = 30;
//    stereo->parameters()->ctf.warps  = 5;
//    stereo->parameters()->ctf.apply_median_filter = true;
//    stereo->parameters()->lambda = 20;

//    stereo->addImage(im0);
//    stereo->addImage(im1);

//    imp::cu::ImageGpu32fC1::Ptr cu_mu = std::make_shared<imp::cu::ImageGpu32fC1>(im0->size());
//    imp::cu::ImageGpu32fC1::Ptr cu_sigma2 = std::make_shared<imp::cu::ImageGpu32fC1>(im0->size());
//    cu_mu->setValue(-5.f);
//    cu_sigma2->setValue(0.0f);

//    stereo->setFundamentalMatrix(F_mov_fix);
//    stereo->setIntrinsics({cu_cam, cu_cam});
//    stereo->setExtrinsics(T_mov_fix);
//    stereo->setDepthProposal(cu_mu, cu_sigma2);

//    stereo->solve();

//    std::shared_ptr<imp::cu::ImageGpu32fC1> d_disp = stereo->getDisparities();


//    {
//      imp::Pixel32fC1 min_val,max_val;
//      imp::cu::minMax(*d_disp, min_val, max_val);
//      std::cout << "disp: min: " << min_val.x << " max: " << max_val.x << std::endl;
//    }

//    imp::cu::cvBridgeShow("im0", *im0);
//    imp::cu::cvBridgeShow("im1", *im1);
////    *d_disp *= -1;
//    {
//      imp::Pixel32fC1 min_val,max_val;
//      imp::cu::minMax(*d_disp, min_val, max_val);
//      std::cout << "disp: min: " << min_val.x << " max: " << max_val.x << std::endl;
//    }

//    imp::cu::cvBridgeShow("disparities", *d_disp, -18.0f, 11.0f);
//    imp::cu::cvBridgeShow("disparities minmax", *d_disp, true);
//    cv::waitKey();
//  }
//  catch (std::exception& e)
//  {
//    std::cout << "[exception] " << e.what() << std::endl;
//    assert(false);
//  }

  return EXIT_SUCCESS;

}

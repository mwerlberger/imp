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

imp::ImageCv32fC1::Ptr loadUint4ToFloat(const std::string& filename)
{
  cv::Mat im_as_4uint = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat im_32f(im_as_4uint.rows, im_as_4uint.cols/4, CV_32F, im_as_4uint.data);
  imp::ImageCv32fC1::Ptr img(new imp::ImageCv32fC1(im_32f.clone()));
  return img;
  //img.reset(new imp::ImageCv32fC1(im_32f.clone()));
}

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

  imp::ImageCv32fC1::Ptr image_ref;
  imp::ImageCv32fC1::Ptr image_mov;
  imp::cu::ImageGpu32fC1::Ptr cu_image_ref;
  imp::cu::ImageGpu32fC1::Ptr cu_image_mov;
  Eigen::Quaterniond q_world_ref(0.0, 0.0, 0.0, 0.0);
  Eigen::Vector3d t_world_ref(0.0, 0.0, 0.0);
  Eigen::Quaterniond q_world_mov(0.0, 0.0, 0.0, 0.0);
  Eigen::Vector3d t_world_mov(0.0, 0.0, 0.0);

  imp::cu::PinholeCamera cu_cam(414.090181, 413.798838, 355.566916, 246.337069);

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
    std::stringstream ss_img_name;
    ss_img_name << input_directory << "/" << img_id << "_image_.png";

    //
    // read all images
    //

    // ref / mov image
    std::cout << "reading " << ss_img_name.str() << std::endl;
    if (!image_ref || img_id == 0)
    {
      q_world_ref = q_world_cur;
      t_world_ref = t_world_cur;
      imp::cvBridgeLoad(image_ref, ss_img_name.str(), imp::PixelOrder::gray);
      cu_image_ref.reset(new imp::cu::ImageGpu32fC1(*image_ref));
      continue;
    }
    q_world_mov = q_world_cur;
    t_world_mov = t_world_cur;
    imp::cvBridgeLoad(image_mov, ss_img_name.str(), imp::PixelOrder::gray);
    cu_image_mov.reset(new imp::cu::ImageGpu32fC1(*image_mov));
    ss_img_name.str("");

    // mu
    ss_img_name << input_directory << "/" << img_id << "_mu_.png";
    std::cout << "reading " << ss_img_name.str() << std::endl;
    imp::ImageCv32fC1::Ptr mu = loadUint4ToFloat(ss_img_name.str());
    //loadUint4ToFloat(mu, ss_img_name.str());
    imp::cvBridgeShow("mu", *mu, true);
    imp::cu::ImageGpu32fC1::Ptr cu_mu(new imp::cu::ImageGpu32fC1(*mu));
    ss_img_name.str("");

    // a
    ss_img_name << input_directory << "/" << img_id << "_a_.png";
    std::cout << "reading " << ss_img_name.str() << std::endl;
    imp::ImageCv32fC1::Ptr a = loadUint4ToFloat(ss_img_name.str());
    imp::cu::ImageGpu32fC1::Ptr cu_a(new imp::cu::ImageGpu32fC1(*a));
    ss_img_name.str("");

    // b
    ss_img_name << input_directory << "/" << img_id << "_b_.png";
    std::cout << "reading " << ss_img_name.str() << std::endl;
    imp::ImageCv32fC1::Ptr b = loadUint4ToFloat(ss_img_name.str());
    imp::cu::ImageGpu32fC1::Ptr cu_b(new imp::cu::ImageGpu32fC1(*b));
    ss_img_name.str("");

    // sigma2
    ss_img_name << input_directory << "/" << img_id << "_sigma2_.png";
    std::cout << "reading " << ss_img_name.str() << std::endl;
    imp::ImageCv32fC1::Ptr sigma2 = loadUint4ToFloat(ss_img_name.str());
    imp::cu::ImageGpu32fC1::Ptr cu_sigma2(new imp::cu::ImageGpu32fC1(*sigma2));
    ss_img_name.str("");

    // mask
    ss_img_name << input_directory << "/" << img_id << "_mask_.png";
    std::cout << "reading " << ss_img_name.str() << std::endl;
    imp::cu::ImageGpu32fC1::Ptr cu_mask;
    imp::cu::cvBridgeLoad(cu_mask, ss_img_name.str(), imp::PixelOrder::gray);
    ss_img_name.str("");


    //
    //
    //
    // im0: fixed image; im1: moving image
    imp::cu::Matrix3f F_fix_mov;
    imp::cu::Matrix3f F_mov_fix;
    Eigen::Matrix3d F_fm, F_mf;
    { // compute fundamental matrix
      Eigen::Matrix3d R_world_mov = q_world_mov.matrix();
      Eigen::Matrix3d R_world_fix = q_world_ref.matrix();
      Eigen::Matrix3d R_fix_mov = R_world_fix.inverse()*R_world_mov;

      // in ref coordinates
      Eigen::Vector3d t_fix_mov = R_world_fix.inverse()*(-t_world_ref + t_world_ref);

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
    imp::cu::SE3<float> T_world_ref(
          static_cast<float>(q_world_ref.w()), static_cast<float>(q_world_ref.x()),
          static_cast<float>(q_world_ref.y()), static_cast<float>(q_world_ref.z()),
          static_cast<float>(t_world_ref.x()), static_cast<float>(t_world_ref.y()),
          static_cast<float>(t_world_ref.z()));
    imp::cu::SE3<float> T_world_mov(
          static_cast<float>(q_world_mov.w()), static_cast<float>(q_world_mov.x()),
          static_cast<float>(q_world_mov.y()), static_cast<float>(q_world_mov.z()),
          static_cast<float>(t_world_mov.x()), static_cast<float>(t_world_mov.y()),
          static_cast<float>(t_world_mov.z()));
    imp::cu::SE3<float> T_mov_ref = T_world_mov.inv() * T_world_ref;
    // end .. compute SE3 transformation

    std::cout << "T_mov_fix:\n" << T_mov_ref << std::endl;


    //
    // run 2-view stereo
    //
    std::unique_ptr<imp::cu::VariationalEpipolarStereo> stereo(
          new imp::cu::VariationalEpipolarStereo());

    imp::cu::ImageGpu32fC1::Ptr cu_disp(new imp::cu::ImageGpu32fC1(cu_image_ref->size()));
    cu_disp->setValue(0.0f);
    imp::cu::ImageGpu32fC1::Ptr cu_mu_proposed(new imp::cu::ImageGpu32fC1(cu_image_ref->size()));
    cu_mu_proposed->setValue(0.0f);

    stereo->parameters()->verbose = 1;
    stereo->parameters()->solver = imp::cu::StereoPDSolver::EpipolarPrecondHuberL1;
    stereo->parameters()->ctf.scale_factor = 0.8f;
    stereo->parameters()->ctf.iters = 30;
    stereo->parameters()->ctf.warps  = 5;
    stereo->parameters()->ctf.apply_median_filter = true;
    stereo->parameters()->lambda = 20;

    stereo->addImage(cu_image_ref);
    stereo->addImage(cu_image_mov);
    imp::cu::ImageGpu32fC1::Ptr tmp = std::make_shared<imp::cu::ImageGpu32fC1>(cu_image_ref->size());
    tmp->setValue(0.0f);

    stereo->setFundamentalMatrix(F_mov_fix);
    stereo->setIntrinsics({cu_cam, cu_cam});
    stereo->setExtrinsics(T_mov_ref);
    stereo->setDepthProposal(cu_mu_proposed, tmp);
    stereo->solve();

    cu_disp = stereo->getDisparities();
    {
      imp::Pixel32fC1 min_val,max_val;
      imp::cu::minMax(*cu_disp, min_val, max_val);
      std::cout << "disp: min: " << min_val.x << " max: " << max_val.x << std::endl;
    }
    //
    // display stuff
    //
#if 0
    imp::cvBridgeShow("image_ref", *image_ref);
    imp::cvBridgeShow("image_mov", *image_mov);
    imp::cvBridgeShow("mu", *mu, true);
    imp::cvBridgeShow("a", *a, true);
    imp::cvBridgeShow("b", *b, true);
    imp::cvBridgeShow("sigma2", *sigma2, true);
    imp::cvBridgeShow("mask", *mask, true);
#endif
#if 1
    imp::cu::cvBridgeShow("cu_image_ref", *cu_image_ref);
    imp::cu::cvBridgeShow("cu_image_mov", *cu_image_mov);
    imp::cu::cvBridgeShow("cu_mu", *cu_mu, true);
    imp::cu::cvBridgeShow("cu_a", *cu_a, true);
    imp::cu::cvBridgeShow("cu_b", *cu_b, true);
    imp::cu::cvBridgeShow("cu_sigma2", *cu_sigma2, true);
    imp::cu::cvBridgeShow("cu_mask", *cu_mask, true);
    imp::cu::cvBridgeShow("var_stereo: cu_disp", *cu_disp, true);
#endif

    cv::waitKey();
  }

  return EXIT_SUCCESS;

}
